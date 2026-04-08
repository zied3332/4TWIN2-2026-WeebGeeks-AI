from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(title="HR AI Similarity Service")

model = SentenceTransformer("all-MiniLM-L6-v2")


class SimilarityRequest(BaseModel):
    activityText: str
    employeeTexts: list[str]


class SimilarityResponse(BaseModel):
    scores: list[float]


def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / denominator)


@app.get("/")
def root():
    return {"message": "HR AI service is running"}


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(payload: SimilarityRequest):
    activity_embedding = model.encode(payload.activityText)

    employee_embeddings = model.encode(payload.employeeTexts)

    scores = [
        max(0.0, min(1.0, cosine_similarity(activity_embedding, emp_embedding)))
        for emp_embedding in employee_embeddings
    ]

    return {"scores": scores}