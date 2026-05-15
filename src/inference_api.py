import io
from PIL import Image
from fastapi.responses import RedirectResponse
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
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

@app.post("/generateNumber", tags=["Generación"])
def generateNumber(number: int):

    generated_tensor = app.state.model.generate_number(number)
    img_array = (generated_tensor * 255).astype("uint8")
    img = Image.fromarray(img_array)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    return Response(content=byte_im, media_type="image/png")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")