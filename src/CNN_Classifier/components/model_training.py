from CNN_Classifier.config.configuration import TrainingConfig
from pathlib import Path
import tensorflow as tf


class Training:
    def __init__(self, config:TrainingConfig):
        self.config = config
        self.validation_ratio = 0.2


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train_valid_generator(self):
        self.train_data = tf.keras.utils.image_dataset_from_directory(
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
            subset="training",
            interpolation="bilinear",
            verbose=True,
        )

    def train(self):
        
        self.model.fit(
            self.train_data,
            epochs=self.config.params_epochs,
            shuffle=True,
            batch_size=self.config.params_batch_size,
        )
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )