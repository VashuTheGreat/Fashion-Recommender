import asyncio
import logging
from pathlib import Path
from fashion_recommender.pipeline.prediction_pipeline import PredictionPipeline
from fashion_recommender.constants import ROOT_DIR

# Setup basic logging for standalone execution
logging.basicConfig(level=logging.INFO)

async def main():
    print("--- Starting Prediction Pipeline Test ---")
    
    # Use the test image from root
    img_path = ROOT_DIR / "test.jpg"
    if not img_path.exists():
        print(f"Error: Test image not found at {img_path}")
        return

    pipeline = PredictionPipeline()
    
    print(f"Loading pre-trained models...")
    try:
        pipeline.extractor.load()
        pipeline.recommender.load()
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Hint: Run the training test first to generate artifacts.")
        return
    
    print(f"Running prediction on {img_path.name}...")
    links, styles = await pipeline.predict(img_path)
    
    print(f"Prediction Successful!")
    print(f"Found {len(links)} recommendations:")
    for i, link in enumerate(links):
        print(f"  {i+1}: {link}")

if __name__ == "__main__":
    asyncio.run(main())
