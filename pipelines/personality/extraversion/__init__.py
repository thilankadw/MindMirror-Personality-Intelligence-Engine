"""Package initialization for pipelines.personality.extraversion."""
# # Pipeline package
# """Weekly State Prediction Pipelines."""

# from pipelines.data_pipeline import data_pipeline
# from pipelines.train_pipeline import training_pipeline
# from pipelines.inference_pipeline import run_inference_pipeline

# __all__ = ['data_pipeline', 'training_pipeline', 'run_inference_pipeline']

# def run_all():
#     """Run all pipelines in sequence."""
#     import logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     logger = logging.getLogger(__name__)
    
#     logger.info("="*60)
#     logger.info("Step 1: Running data pipeline")
#     logger.info("="*60)
#     data_pipeline()
    
#     logger.info("\n" + "="*60)
#     logger.info("Step 2: Running training pipeline")
#     logger.info("="*60)
#     from utils.config import get_model_config
#     model_config = get_model_config()
#     training_pipeline(model_params=model_config.get('model_params'))
    
#     logger.info("\n" + "="*60)
#     logger.info("Step 3: Running inference pipeline")
#     logger.info("="*60)
#     import pandas as pd
#     df = pd.read_json('data/raw/reddit_post_data.json').head(5)
#     run_inference_pipeline(input_data=df)
    
#     logger.info("\n" + "="*60)
#     logger.info("All pipelines completed successfully!")
#     logger.info("="*60)
