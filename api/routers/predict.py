import logging
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fashion_recommender.utils import save_upload, delete_file
from api.dependencies import get_pipeline, PredictionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("")
async def predict(
    file: UploadFile = File(...),
    pipeline: PredictionPipeline = Depends(get_pipeline),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported")

    logger.info(f"Received upload: {file.filename} ({file.content_type})")
    contents = await file.read()
    suffix = Path(file.filename).suffix or ".jpg"
    img_path = await save_upload(contents, suffix)

    try:
        result, json_data = await pipeline.predict(img_path)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await delete_file(img_path)

    return {"result": result, "json_data": json_data}
