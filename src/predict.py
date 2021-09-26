from src.model import Model
import src.config as config
from src.transforms import valid_transforms


def get_prediction(img):
    img = valid_transforms(img).unsqueeze(0).to("cpu")

    model = Model()
    model.load_weights(config.MODEL_PATH)
    model.eval()
    model.to("cpu")

    output = model(img).detach()
    pred_idx = output.numpy().argmax()

   
    return config.PRED_TO_CLASS[pred_idx]
