from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
from src.inference import predict_image
import io

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: int
    tax: int = None

@app.post('/items')
def add_item(item: Item):
    return {"item": item}

@app.post('/predict')
async def predict_imagee(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    prediction = predict_image(image)

    return {"prediciton": prediction}