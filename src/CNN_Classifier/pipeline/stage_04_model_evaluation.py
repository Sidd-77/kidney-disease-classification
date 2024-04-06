from CNN_Classifier import logger
import os
from CNN_Classifier.components.model_evaluation import Evaluation
from CNN_Classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation"

class ModelEvalutaionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evalutaion = Evaluation(eval_config)
        evalutaion.evaluation()
        evalutaion.log_into_mlflow()


if __name__=='__main__':
    try:
        os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Sidd-77/kidney-disease-classification.mlflow"
        os.environ['MLFLOW_TRACKING_USERNAME']="Sidd-77"
        os.environ['MLFLOW_TRACKING_PASSWORD']="61bcab158d9ab43c4be35facef8400397b011fcc"
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelEvalutaionPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<")
    except Exception as e:
        raise e

