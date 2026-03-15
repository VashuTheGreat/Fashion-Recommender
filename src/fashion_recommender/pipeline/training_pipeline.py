import logging
import os
import asyncio
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import httpx
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torchvision import models, transforms

from fashion_recommender.constants import (
    IMAGES_CSV_PATH,
    MODEL_PATH,
    NEIGHBORS_MODEL_PATH,
    FILENAMES_PATH,
    IMAGE_TARGET_SIZE,
    KNN_NEIGHBORS,
    KNN_ALGORITHM,
    KNN_METRIC,
    TRAINING_LIMIT,
)

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TrainingPipeline:
    def __init__(self):
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> nn.Module:
        logger.info("Building ResNet50 feature extraction model")
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(base.children())[:-1])
        model.eval()
        model.to(self._device)
        logger.info(f"Model ready on {self._device}")
        return model

    async def _fetch_image_and_extract(self, client: httpx.AsyncClient, url: str) -> np.ndarray | None:
        try:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            x = _TRANSFORM(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(x)
            feat = feat.squeeze().cpu().numpy()
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat
        except Exception as exc:
            logger.error(f"Failed to process image from {url}: {exc}")
            return None

    async def run(self, limit: int = TRAINING_LIMIT) -> None:
        self._model = self._build_model()
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Reading {limit} rows from {IMAGES_CSV_PATH}")
        df = pd.read_csv(IMAGES_CSV_PATH).head(limit)
        
        urls = df['link'].tolist()
        filenames = df['filename'].tolist()
        
        X = []
        valid_filenames = []

        async with httpx.AsyncClient() as client:
            for i, url in enumerate(tqdm(urls, desc="Extracting features on-the-go")):
                feat = await self._fetch_image_and_extract(client, url)
                if feat is not None:
                    X.append(feat)
                    valid_filenames.append(filenames[i])

        X = np.array(X)
        logger.info(f"Feature matrix shape: {X.shape}")

        with open(str(FILENAMES_PATH), "wb") as f:
            pickle.dump(valid_filenames, f)
        logger.info(f"Saved filenames.pkl → {FILENAMES_PATH}")

        knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, algorithm=KNN_ALGORITHM, metric=KNN_METRIC)
        knn.fit(X)
        logger.info("KNN model fitted")

        torch.save(self._model.state_dict(), str(MODEL_PATH))
        logger.info(f"Feature extractor saved → {MODEL_PATH}")

        with open(str(NEIGHBORS_MODEL_PATH), "wb") as f:
            pickle.dump(knn, f)
        logger.info(f"KNN model saved → {NEIGHBORS_MODEL_PATH}")
