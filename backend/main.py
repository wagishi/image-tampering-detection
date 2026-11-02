"""FastAPI application for the VGG19 image tampering classifier."""
from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from inference import predict_image

app = FastAPI(
    title="Image Tampering Detection API",
    description=(
        "Upload a JPEG or PNG image for binary classification by a VGG19 model."
    ),
    version="1.0.0",
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "image-tampering-detection-api"}


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> dict:
    """Classify one image. The filename and MIME type are not trusted alone."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG uploads are supported.",
        )

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {MAX_UPLOAD_BYTES} bytes.",
        )

    try:
        result = await run_in_threadpool(predict_image, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Log the exception server-side in a production service; do not expose
        # implementation details or filesystem paths to the client.
        raise HTTPException(
            status_code=500,
            detail="Image inference failed. Check the server logs.",
        ) from exc

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        **result,
    }
