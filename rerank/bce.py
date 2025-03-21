from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import os
from copy import deepcopy
import concurrent.futures
import torch

class RerankerModel:
    def __init__(self, model_path: str, use_onnx: bool = False, use_cpu: bool = False, use_fp16: bool = False, device: str = None, **kwargs):
        self.use_onnx = use_onnx
        self.use_cpu = use_cpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)
        self.batch_size = kwargs.get('batch_size', 32)
        self.return_tensors = "np" if use_onnx else "pt"
        self.workers = kwargs.get('workers', 4)

        if use_onnx:
            sess_options = SessionOptions()
            sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CPUExecutionProvider'] if use_cpu else [('CUDAExecutionProvider',{
                'device_id': f"{device.split(':')[-1]}" if device is not None and device.startswith('cuda:') else '0'
            }), 'CPUExecutionProvider']
            self.session = InferenceSession(os.path.join(model_path, "rerank.onnx"), sess_options, providers=providers)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
            num_gpus = torch.cuda.device_count()
            self.device = "cuda" if num_gpus > 0 else "cpu" if device is None else 'cuda:{}'.format(int(device)) if device.isdigit() else device
            self.num_gpus = 0 if self.device == "cpu" else 1 if self.device.startswith('cuda:') and num_gpus > 0 else num_gpus
            if use_fp16:
                self.model.half()
            self.model.eval()
            self.model = self.model.to(self.device)
            if self.num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)

    def inference(self, batch):
        if self.use_onnx:
            inputs = {self.session.get_inputs()[0].name: batch['input_ids'], self.session.get_inputs()[1].name: batch['attention_mask']}
            if 'token_type_ids' in batch:
                inputs[self.session.get_inputs()[2].name] = batch['token_type_ids']
            result = self.session.run(None, inputs)
            sigmoid_scores = 1 / (1 + np.exp(-np.array(result[0])))
            return sigmoid_scores.reshape(-1).tolist()
        else:
            with torch.no_grad():
                inputs_on_device = {k: v.to(self.device) for k, v in batch.items()}
                scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(scores)
                return scores.cpu().numpy().tolist()

    def merge_inputs(self, chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(self.tokenizer.sep_token_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    def tokenize_preproc(self, query: str, passages: List[str]):
        query_inputs = self.tokenizer.encode_plus(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 1
        assert max_passage_inputs_length > 10
        overlap_tokens = min(self.overlap_tokens, max_passage_inputs_length * 2 // 7)

        merge_inputs = []
        merge_inputs_idxs = []
        for pid, passage in enumerate(passages):
            passage_inputs = self.tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                if passage_inputs['attention_mask'] is None or len(passage_inputs['attention_mask']) == 0:
                    continue
                qp_merge_inputs = self.merge_inputs(query_inputs, passage_inputs)
                merge_inputs.append(qp_merge_inputs)
                merge_inputs_idxs.append(pid)
            else:
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = self.merge_inputs(query_inputs, sub_passage_inputs)
                    merge_inputs.append(qp_merge_inputs)
                    merge_inputs_idxs.append(pid)

        return merge_inputs, merge_inputs_idxs

    def rerank(self, query: str, passages: List[str]):
        tot_batches, merge_inputs_idxs_sort = self.tokenize_preproc(query, passages)
        tot_scores = []

        if self.use_onnx:
            tot_scores = self._rerank_onnx(tot_batches)
        else:
            tot_scores = self._rerank_torch(tot_batches)

        merge_tot_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(merge_inputs_idxs_sort, tot_scores):
            merge_tot_scores[pid] = max(merge_tot_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_tot_scores)[::-1]
        sorted_passages = []
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_tot_scores[mid])
            sorted_passages.append(passages[mid])
        
        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores,
            'rerank_ids': merge_scores_argsort.tolist()
        }

    def _rerank_onnx(self, tot_batches):
        tot_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self.tokenizer.pad(tot_batches[k:k + self.batch_size], padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=self.return_tensors)
                future = executor.submit(self.inference, batch)
                futures.append(future)
            for future in futures:
                scores = future.result()
                tot_scores.extend(scores)
        return tot_scores

    def _rerank_torch(self, tot_batches):
        tot_scores = []
        with torch.no_grad():
            for k in range(0, len(tot_batches), self.batch_size):
                batch = self.tokenizer.pad(tot_batches[k:k + self.batch_size], padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=self.return_tensors)
                scores = self.inference(batch)
                tot_scores.extend(scores)
        return tot_scores