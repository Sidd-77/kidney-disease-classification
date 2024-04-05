import tensorflow as tf
import os
from pathlib import Path
import mlflow
import mlflow.keras 
from urllib.parse import urlparse
from CNN_Classifier.config.configuration import EvalutationConfig
from CNN_Classifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvalutationConfig):
        self.config = config
        self.validation_ratio=0.2

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def _valid_generator(self):
        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
                self.config.training_data,
                labels="inferred",
                label_mode="categorical",
                class_names=['Cyst', 'Normal', 'Stone', 'Tumor'],
                color_mode="rgb",
                batch_size=self.config.params_batch_size,
                image_size=self.config.params_image_size[:-1],
                shuffle=True,
                seed=69,
                validation_split=self.validation_ratio,
                subset="validation",
                interpolation="bilinear",
                verbose=True,
        )

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != 'file':
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")