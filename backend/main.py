from fastapi import FastAPI
from fastapi.responses import Response
from pathlib import Path


from index import top_k_images

image_dir = Path(__file__).parent
app = FastAPI()


@app.get("/api/search")
async def root(query: str):
    return Response(top_k_images(query))
