from fashion_recommender.components.feature_extractor import FeatureExtractor
from fashion_recommender.components.recommender import Recommender
from fashion_recommender.pipeline.prediction_pipeline import PredictionPipeline

_pipeline: PredictionPipeline | None = None


def get_pipeline() -> PredictionPipeline:
    global _pipeline
    if _pipeline is None:
        extractor = FeatureExtractor()
        extractor.load()
        recommender = Recommender()
        recommender.load()
        _pipeline = PredictionPipeline()
    return _pipeline
