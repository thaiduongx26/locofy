import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException

from inference import ServingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model_serving = ServingModel()

@app.post("/inference")
def inference_endpoint(data: Dict[str, Any]):
    try:
        res = model_serving.single_predict(data)
        return {"output": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Wrong input format.")
