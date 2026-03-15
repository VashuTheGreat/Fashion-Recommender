import logging
from fashion_recommender.constants import (
    MODEL_PATH,
    NEIGHBORS_MODEL_PATH,
    FILENAMES_PATH,
    UPLOAD_DIR,
)
import asyncio
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


def ensure_dirs() -> None:
    for d in [UPLOAD_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Required artifact directories verified")


async def save_upload(contents: bytes, suffix: str = ".jpg") -> Path:
    ensure_dirs()
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, dest.write_bytes, contents)
    logger.debug(f"Saved uploaded file → {dest}")
    return dest


async def delete_file(path: Path) -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, path.unlink, True)
        logger.debug(f"Deleted temp file → {path}")
    except Exception as exc:
        logger.warning(f"Could not delete {path}: {exc}")


def verify_artifacts() -> None:
    missing = []
    for p in [MODEL_PATH, NEIGHBORS_MODEL_PATH, FILENAMES_PATH]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        logger.warning(f"Missing artifact files: {missing}")
    else:
        logger.info("All required artifact files found")
