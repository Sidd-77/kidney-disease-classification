from CNN_Classifier.config.configuration import TrainingConfig
from pathlib import Path
import tensorflow as tf


class Training:
    def __init__(self, config:TrainingConfig):
        self.config = config
        self.validation_ratio = 0.2


    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(self.model.output)
        prediction = tf.keras.layers.Dense(
            units=4,
            activation="softmax"
        )(flatten_in)

        self.full_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=prediction
        )

        opt = tf.keras.optimizers.RMSprop(
            learning_rate=self.config.params_learning_rate,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=100,
            name="rmsprop",
        )

        self.full_model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        self.full_model.summary()
        


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
        self.full_model.summary()
        self.full_model.fit(
            self.train_data,
            epochs=self.config.params_epochs,
            shuffle=True,
            batch_size=self.config.params_batch_size,
        )
        self.save_model(
            path=self.config.trained_model_path,
            model=self.full_model
        )