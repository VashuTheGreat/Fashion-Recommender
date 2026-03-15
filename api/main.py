import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

import logger as _logger_setup
from api.routers import predict
from api.dependencies import get_pipeline

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Fashion Recommender API starting up — loading models")
    get_pipeline()
    log.info("Models loaded. API ready.")
    yield
    log.info("Fashion Recommender API shutting down")


app = FastAPI(
    title="Fashion Recommender API",
    description="Upload a fashion image and get similar product recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok"}
