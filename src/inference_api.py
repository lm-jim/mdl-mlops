import io
from PIL import Image
import torch
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from src import utils

class GenerateRequest(BaseModel):
    number: int = Field(ge=0, le=9, description="El número a generar (0-9)")

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n\n\n====================\n\n.       INICIO API REST      \n")
    
    app.state.model = utils.load_best_model()

    yield  # <- aquí la API queda disponible
    # Empieza el teardown

    print("\n\n\n====================\n\n.       FIN API REST      \n")


app = FastAPI(
    title="Generador de Imágenes MNIST",
    lifespan=lifespan
)

@app.get("/")
def saluda():
    return {"message" : "Utiliza POST /generate con JSON {\"number\": int} para generar una imagen"}

@app.post("/generate")
def predict(number: int):

    generated_tensor = app.state.model.generate_number(number)
    img_array = (generated_tensor * 255).astype("uint8")
    img = Image.fromarray(img_array)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    return Response(content=byte_im, media_type="image/png")