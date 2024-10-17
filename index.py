from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: int
    tax: int = None

@app.post('/items')
def add_item(item: Item):
    return {"item": item}
