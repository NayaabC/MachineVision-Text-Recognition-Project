import cv2
import typing
import numpy as np

from mltu.inferenceModel import OmnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OmnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list
    
    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        predictions = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(predictions, self.char_list)[0]

        return text
    
if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configPath = r"C:\Users\NC\Documents\Rutgers\Grad\Machine Vision\Project\Models\03_handwriting_recognition\202311091714\configs.yaml"
    configs = BaseModelConfigs.load(configPath)

    textPattern = configs.vocab
    modelPath = configs.model_path

    model = ImageToWordModel(model_path=modelPath, char_list=textPattern)

    




    