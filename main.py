from CNN_Classifier import logger
import os
from CNN_Classifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CNN_Classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from CNN_Classifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from CNN_Classifier.pipeline.stage_04_model_evaluation import ModelEvalutaionPipeline

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Sidd-77/kidney-disease-classification.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="Sidd-77"
os.environ['MLFLOW_TRACKING_PASSWORD']="61bcab158d9ab43c4be35facef8400397b011fcc"

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    raise e


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
except Exception as e:
    raise e


STAGE_NAME = "Model Training"

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        raise e
    

STAGE_NAME = "Model Evaluaion"

if __name__=='__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelEvalutaionPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        raise e