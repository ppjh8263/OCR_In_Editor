from io import BytesIO
import os

import torch

from PIL import Image
import numpy as np
from torchvision import transforms
from src.model import Model

model = None

CLASSES = ["Metal", "Paper", "Paperpack", "Plastic", "Plasticbag","Styrofoam",]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trsf=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]
    )

def load_model(model_dir='/opt/ml/ocrAPI/src/latest',weight_name="best.pt"):
    weight = os.path.join(model_dir, weight_name)
    model_config = os.path.join(model_dir, "model.yml")

    model_instance = Model(model_config, verbose=True)
    model_instance.model.load_state_dict(
        torch.load(weight, map_location=torch.device("cpu"))
    )
    model = model_instance.model
    print("Model loaded")
    return model

@torch.no_grad()
def inference(img: Image.Image):
    global model
    if model is None:
        model = load_model()

    model = model.to(device)
    model.eval()
    img = img.to(device)
    pred = model(img)
    confidence = torch.max(pred)
    class_idx = torch.argmax(pred)
    response = {}
    response["result"] = CLASSES[class_idx]
    response["confidence"] = confidence.item()
    return response


def predict(image: Image.Image):
    # global model
    # if model is None:
    #     model = load_model()
    img_size=image.size
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    img_resize=image.shape
    # result = decode_predictions(model.predict(image), 2)[0]
    result=[[img_size,img_resize,0.5],]

    response = []
    for i, res in enumerate(result):
        resp = {}
        resp["img_size"] = res[0]
        resp["img_resize"] = res[1]
        resp["confidence"] = f"{res[2]*100:0.2f} %"
        # resp["class"] = res[1]
        # resp["confidence"] = f"{res[]*100:0.2f} %"

        response.append(resp)

    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    print(f'IMAGE SIZE: {image.size}')
    image.save('test.jpg')
    image = trsf(image)
    image = image.reshape([1, 3, 224, 224])
    return image