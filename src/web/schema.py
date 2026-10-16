from pydantic import BaseModel

# 定义商品类
class Item(BaseModel):
    name: str
