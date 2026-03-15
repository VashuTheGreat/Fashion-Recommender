import logging
import asyncio
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from fashion_recommender.constants import MODEL_PATH, IMAGE_TARGET_SIZE

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def _build_model(self) -> nn.Module:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(base.children())[:-1])
        model.eval()
        model.to(self._device)
        return model

    def load(self) -> None:
        if self._model is not None:
            return
        if MODEL_PATH.exists():
            logger.info(f"Loading saved feature extractor from {MODEL_PATH}")
            self._model = self._build_model()
            self._model.load_state_dict(torch.load(str(MODEL_PATH), map_location=self._device))
        else:
            logger.info("No saved model found — building from ImageNet weights")
            self._model = self._build_model()
        logger.info(f"Feature extractor ready on {self._device}")

    def _extract_sync(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        x = _TRANSFORM(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(x)
        feat = feat.squeeze().cpu().numpy()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat

    async def extract(self, img_path: Path) -> np.ndarray:
        if self._model is None:
            self.load()
        loop = asyncio.get_running_loop()
        features = await loop.run_in_executor(None, self._extract_sync, str(img_path))
        logger.debug(f"Features extracted for {img_path.name}")
        return features
