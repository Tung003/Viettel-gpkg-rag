# from module_RAG import RAG_vi_mrc_large     # run base dir
from app.module_RAG import RAG_vi_mrc_large 
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

import logging
import os
import asyncio
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor= ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="VIETNAMESE LLM", version="1.0.0")

BASE_DIR=os.path.dirname(os.path.abspath(__file__)) # run base dir app
templates = Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))   # run base dir

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

RAG_pipeline=None
@app.on_event("startup")
async def startup():
    global RAG_pipeline
    try:
        logger.info("Starting RAG system...")
        vectorsstore_path = os.path.join(BASE_DIR, "vectors_store")
        RAG_pipeline = RAG_vi_mrc_large( name_model_llm = os.path.join(BASE_DIR,"models","vi-mrc-large"),
                                         name_model_embedding = os.path.join(BASE_DIR,"models","vietnamese-sbert"),
                                         name_model_find_relevant = os.path.join(BASE_DIR,"models","paraphrase-multilingual-MiniLM-L12-v2"),
                                         data_path = [os.path.join(BASE_DIR,"data","train"),
                                                      os.path.join(BASE_DIR,"data","processed")],
                                         vectors_store_path = vectorsstore_path,)

        # if not os.path.exists(vectorsstore_path):
        #     print("chưa có folder")
        #     RAG_pipeline.build_vectors_store()

        logger.info("RAG system successfully...")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_initialized": RAG_pipeline is not None}

@app.post("/ask")
async def ask_question(req: Question):
    try:
        if not RAG_pipeline:
            raise HTTPException(status_code=503, detail="RAG system not initialized")

        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty")

        logger.info(f"Received question: {req.question}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, RAG_pipeline.ask , req.question)

        logger.info("Answer generated successfully")
        return {
            "question": req.question,
            "answer": result["answer"],
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ask: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/test")
async def test_endpoint():
    return {"message": "API is working!", "timestamp": "2024-01-01"}

# if __name__ == "__main__":
#     uvicorn.run(
#         "RAG_API:app",
#         host="127.0.0.1",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )
