
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import *

import torch
from pathlib import Path
from io import BytesIO
import sys
import aiohttp
import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


learner = load_learner(".")
app = Starlette()


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    losses = img.predict(learner)
    return JSONResponse({
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8080)


# BUCKET_NAME = "cognitivehw"
# REGION_NAME = "us-east-1"
# S3_AWS_ACCESS_KEY_ID = "ASIA32YEBZGQXM2PGBIK"
# S3_AWS_SECRET_ACCESS_KEY = "OnfUFUqKS7tBbCJnjX4Zw3GOCm2HVG6+uAbUxtTO"
# S3_AWS_SESSION_TOKEN = "FwoGZXIvYXdzEAgaDIL3yqWJhI201oRQ+iLGAa0BXtXDmpHk1B5rzoQodmH2KpjzrDeIz7dnyYxThMAY1Nt0MTlPLmO5jqkjO/yOmoVjRWK7sBiQiMcYQwQsPFeuVOaTg4GxC8limvWt872NZX9PfIsPR/pZvthuRBTCSCpvK9AOK7YLQdL2CDGD8FNaYLGyzsICSoKfVKaCEbXg7UN6KmihT7G7QJE6q9y7gOtru+Q9wGoygmGRsgFDLP9DecFRx/KLR5rkf1kJvqnWSaLEDbY70QXx/A9oIfF1kGVTTtsPmyiojKfyBTItB1h8npHc3pLJyKPc9i0r0nAY94Atw6+CDGyychMo8zmzPZtOUXlGbqwsEb4D"


# app.add_middleware(
#     S3StorageMiddleware,
#     bucket_name=BUCKET_NAME,
#     region_name=REGION_NAME,
#     aws_access_key_id=S3_AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=S3_AWS_SECRET_ACCESS_KEY,
#     aws_session_token=S3_AWS_SESSION_TOKEN,
#     static_dir="static",
# )
# handler = Mangum(app)
