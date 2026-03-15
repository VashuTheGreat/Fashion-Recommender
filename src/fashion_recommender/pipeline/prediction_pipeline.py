import logging
from pathlib import Path

from fashion_recommender.components.feature_extractor import FeatureExtractor
from fashion_recommender.components.recommender import Recommender

logger = logging.getLogger(__name__)


class PredictionPipeline:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.recommender = Recommender()

    async def predict(self, img_path: Path) -> tuple[list[str], list[str]]:
        logger.info(f"Prediction pipeline started for {img_path.name}")
        features = await self.extractor.extract(img_path)
        links, styles = await self.recommender.recommend(features)
        logger.info(f"Prediction pipeline complete — {len(links)} results")
        return links, styles
