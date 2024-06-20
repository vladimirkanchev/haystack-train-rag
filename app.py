from haystack import Pipeline

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import json

from dotenv import load_dotenv, find_dotenv

import box
import yaml

from src.inference import run_pipeline
from src.rag_pipelines import select_rag_pipeline



load_dotenv(find_dotenv())

with open('./src/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

print("Import Successfully")
app = FastAPI()
# Configure templates
templates = Jinja2Templates(directory="templates")

def get_result(query: str):
    """ """
    rag_pipeline = select_rag_pipeline()
    rag_answer, retrieved_docs = run_pipeline(query, rag_pipeline)

    return rag_answer, retrieved_docs


@app.get("/")
async def index(request: Request):
    """ """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(...)):
    print(question)
    answer, relevant_documents = get_result(question)
    print(relevant_documents)
    response_data = jsonable_encoder(json.dumps(
        {"answer": answer,
         "relevant_documents": relevant_documents
         }
         )
        )
    res = Response(response_data)
    return res

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)