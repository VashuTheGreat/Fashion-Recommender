# Fashion Recommender

A deep learning-based fashion product recommendation system using **ResNet50** feature extraction (PyTorch) and **KNN** similarity search, served via a production-grade **FastAPI** async backend.

---

## Project Structure

```
Fashion-Recommender/
├── src/
│   └── fashion_recommender/
│       ├── constants/         # All paths & hyperparameters
│       ├── utils/             # Async file I/O helpers
│       ├── components/
│       │   ├── feature_extractor.py   # Singleton ResNet50 extractor
│       │   └── recommender.py         # Singleton KNN recommender
│       ├── pipeline/
│       │   ├── training_pipeline.py   # Online image training pipeline
│       │   └── prediction_pipeline.py # Orchestrate inference
│       └── tests/
│           ├── run_train.py           # Standalone training test
│           └── run_pred.py            # Standalone prediction test
├── api/
│   ├── main.py                # FastAPI app with lifespan
│   ├── dependencies.py        # Dependency injection
│   ├── templates/             # HTML UI
│   └── routers/
│       └── predict.py         # POST /predict endpoint
├── logger/                    # Rotating file + console logger
├── exception/                 # Custom exception with traceback
├── data/
│   └── images.csv             # Source data with image URLs
├── artifacts/                 # Generated models & filenames (gitignored)
├── pyproject.toml             # uv / hatchling project config
├── main.py                    # Main server entrypoint
└── Dockerfile
```

---

## Setup with `uv`

```bash
uv sync
```

---

## Usage

### 1. Training (Online "On-The-Go")
The training pipeline fetches images directly from URLs in `data/images.csv`. By default, it trains on the first `TRAINING_LIMIT` items (set in `constants/__init__.py`).

**Run training:**
```bash
uv run python src/fashion_recommender/tests/run_train.py
```

### 2. Running the Server
Once artifacts are generated, start the FastAPI server:

```bash
uv run main.py
```
Open `http://localhost:8000` to access the web UI.

### 3. Standalone Prediction Test
Verify the pipeline with a local image:

```bash
uv run python src/fashion_recommender/tests/run_pred.py
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/`  | Web UI |
| `POST` | `/predict` | Upload image → get similar recommendations |
| `GET`  | `/health` | Health check |

---

## Core Technologies
- **Backend**: FastAPI, Uvicorn
- **ML Framework**: PyTorch, Torchvision
- **Search**: Scikit-learn (KNN)
- **Data**: Pandas, PIL, HTTPX
- **Management**: uv (Dependency Management)
