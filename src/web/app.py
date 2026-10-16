import torch
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config.config import *
from web.schema import Item, Result
from runner.predict import Predictor
import uvicorn

# 创建app对象
app = FastAPI()

# 1. 定义设备
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
# 2. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. 加载模型
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR/'best')

# 定义预测器
predictor = Predictor(
    model=model,
    tokenizer=tokenizer,
    device=device
)

@app.post("/predict", response_model=Result)
def predict(item: Item):
    category = predictor.predict(item.name)
    return Result(category=category)

def run_app():
    uvicorn.run(app=app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_app()
