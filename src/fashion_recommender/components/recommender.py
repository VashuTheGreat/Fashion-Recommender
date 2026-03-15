import logging
import asyncio
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from fashion_recommender.constants import (
    NEIGHBORS_MODEL_PATH,
    FILENAMES_PATH,
    IMAGES_CSV_PATH,
    STYLES_DIR,
)

logger = logging.getLogger(__name__)


class Recommender:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._neighbors = None
            cls._instance._filenames = None
            cls._instance._data = None
        return cls._instance

    def load(self) -> None:
        if self._neighbors is not None:
            return
        logger.info("Loading KNN neighbors model and metadata")
        with open(str(NEIGHBORS_MODEL_PATH), "rb") as f:
            self._neighbors = pickle.load(f)
        with open(str(FILENAMES_PATH), "rb") as f:
            self._filenames = pickle.load(f)
        self._data = pd.read_csv(str(IMAGES_CSV_PATH))
        logger.info(f"Loaded {len(self._filenames)} filenames and CSV with {len(self._data)} rows")

    def _get_image_link(self, file_idx: int) -> str:
        fname = self._filenames[file_idx]
        base = Path(fname).name if "\\" in fname or "/" in fname else fname
        row = self._data[self._data["filename"] == base]["link"]
        if row.empty:
            logger.warning(f"No link found for filename index {file_idx} → {base}")
            return ""
        return row.iloc[0]

    def _get_style_json(self, file_idx: int) -> str:
        fname = self._filenames[file_idx]
        base = Path(fname).name if "\\" in fname or "/" in fname else fname
        stem = Path(base).stem
        style_path = STYLES_DIR / f"{stem}.json"
        if style_path.exists():
            return style_path.read_text(encoding="utf-8")
        logger.warning(f"Style JSON not found: {style_path}")
        return "{}"

    def _recommend_sync(self, features: np.ndarray) -> tuple[list[str], list[str]]:
        distances, indices = self._neighbors.kneighbors([features])
        links = []
        styles = []
        for idx in indices[0]:
            links.append(self._get_image_link(idx))
            styles.append(self._get_style_json(idx))
        return links, styles

    async def recommend(self, features: np.ndarray) -> tuple[list[str], list[str]]:
        if self._neighbors is None:
            self.load()
        loop = asyncio.get_running_loop()
        links, styles = await loop.run_in_executor(None, self._recommend_sync, features)
        logger.info(f"Returning {len(links)} recommendations")
        return links, styles
