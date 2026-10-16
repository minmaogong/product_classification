from fastapi import FastAPI
from web.schema import Item

# 创建app对象
app = FastAPI()

@app.post("/predict")
def predict(item: Item):
    return {"name": item.name}
