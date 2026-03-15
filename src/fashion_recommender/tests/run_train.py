import asyncio
import logging
from fashion_recommender.pipeline.training_pipeline import TrainingPipeline

# Setup basic logging for standalone execution
logging.basicConfig(level=logging.INFO)

async def main():
    print("--- Starting Training Pipeline Test ---")
    pipeline = TrainingPipeline()
    await pipeline.run()
    print("--- Training Pipeline Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main())
