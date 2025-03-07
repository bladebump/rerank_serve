from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
from rerank.bce import RerankerModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置文件示例（可用环境变量替代）
class Config:
    MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/data/model/jina-reranker-v2-base-multilingual/")
    USE_ONNX = os.getenv("USE_ONNX", "True") == "True"
    USE_CPU = os.getenv("USE_CPU", "False") == "True"

app = FastAPI(title="BCE Embedding Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Embedding Agent
rerank_agent = None

class RerankRequest(BaseModel):
    query: str
    passages: List[str]

class RerankResponse(BaseModel):
    rerank_ids: List[int]
    rerank_scores: List[float]
    rerank_passages: List[str]

@app.on_event("startup")
async def startup_event():
    global rerank_agent
    try:
        rerank_agent = RerankerModel(
            model_path=Config.MODEL_PATH,
            use_cpu=Config.USE_CPU,
            use_onnx=Config.USE_ONNX,
            device="cuda:1"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding agent: {str(e)}")

@app.post("/rerank", response_model=RerankResponse)
async def get_embeddings(request: RerankRequest):
    logger.info(f"Received request: {request}")
    if rerank_agent is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # 获取嵌入
        reranks_answer = rerank_agent.rerank(
            query=request.query,
            passages=request.passages
        )
        
        return {
            "rerank_ids": reranks_answer["rerank_ids"],
            "rerank_scores": reranks_answer["rerank_scores"],
            "rerank_passages": reranks_answer["rerank_passages"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": rerank_agent is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9239)