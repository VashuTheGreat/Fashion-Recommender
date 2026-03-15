from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT_DIR / "artifacts" / "features_extractor.pt"
NEIGHBORS_MODEL_PATH = ROOT_DIR / "artifacts" / "neighbors_model.pkl"
FILENAMES_PATH = ROOT_DIR / "artifacts" / "filenames.pkl"
IMAGES_CSV_PATH = ROOT_DIR / "data" / "images.csv"
STYLES_DIR = ROOT_DIR / "artifacts" / "styles"
IMAGES_DIR = ROOT_DIR / "artifacts" / "images" / "images"
IMAGES_FEATURES_DIR = ROOT_DIR / "artifacts" / "images_features"
UPLOAD_DIR = ROOT_DIR / "artifacts" / "uploads"

IMAGE_TARGET_SIZE = (224, 224)
KNN_NEIGHBORS = 6
KNN_ALGORITHM = "brute"
KNN_METRIC = "euclidean"
TRAINING_LIMIT = 1000
