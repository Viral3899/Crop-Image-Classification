import tensorflow as tf
from Crop_Image_Classification.constants import *
from Crop_Image_Classification.config.configuration import EvaluationConfig
from Crop_Image_Classification.utils.common import save_json
from datetime import datetime


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    @staticmethod
    def get_current_time_stamp():
        return f"{datetime.now().strftime('%Y%m%d%H')}"
    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        print(self.score)

    
    def save_score(self):
        scores = {f"loss_{Evaluation.get_current_time_stamp()}": self.score[0], f"accuracy_{Evaluation.get_current_time_stamp()}": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
