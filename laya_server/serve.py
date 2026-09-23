"""Local HTTP wrapper around laya-mlx for layatrade-rs.

Run with: uvicorn serve:app --host 127.0.0.1 --port 8787
"""
import laya_mlx as laya
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
agent = laya.load("aac6fef/laya-mlx")

TRADE_QUESTION = {
    "should_trade": {
        "type": "noul",
        "instructions": (
            "Given this order book state, is now a good moment to execute "
            "a trade in the configured direction?"
        ),
    }
}


class PredictRequest(BaseModel):
    state: str


class PredictResponse(BaseModel):
    confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    result = agent.predict(req.state, TRADE_QUESTION)
    confidence = float(result["answers"]["should_trade"]["noul"])
    return PredictResponse(confidence=confidence)
