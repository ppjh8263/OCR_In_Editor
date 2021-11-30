import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from typing import List
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from classification import predict, read_imagefile, inference
import base64
import io
from PIL import Image
from docs import description
import datetime
import random
from papago import translation_en2ko
from generate_point import get_random_polygon, get_random_point

app = FastAPI(description=description)

@app.get("/")
def read_root():
    return "Boost Camp AI tech CV7's API"


@app.post("/classification/image")
async def predict_api_image(file: UploadFile = File(...)):
    time_start = time.monotonic()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = inference(image)
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@app.post("/classification/base64")
async def predict_api_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))

    prediction = inference(image)
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@app.post("/bbox_demo/image")
async def demo_image(file: UploadFile = File(...)):
    time_start = time.monotonic()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print(datetime.datetime.now())
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    print("success image upload!!")
    prediction=[random.randint(0,5)]
    for idx in range(prediction[0]):
        str_trns=translation_en2ko(f"demo trasnslation of bbox{idx}")
        print(f'Papago responese : {str_trns}')
        if str_trns[0]==200:
            prediction.append(
                {
                    'translation':str_trns[1],
                    'point':get_random_polygon()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{str_trns[0]}]...',
                    'point':[]}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@app.post("/bbox_demo/base64")
async def demo_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))
    print(type(image))
    print("success image upload!!")
    prediction=[random.randint(0,5)]
    for idx in range(prediction[0]):
        str_trns=translation_en2ko(f"demo trasnslation of bbox{idx}")
        print(f'Papago responese : {str_trns}')
        if str_trns[0]==200:
            prediction.append(
                {
                    'translation':str_trns[1],
                    'point':get_random_polygon()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{str_trns[0]}]...',
                    'point':[]}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@app.post("/bbox_demo/nopapago")
async def demo_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))
    print("success image upload!!")
    prediction=[random.randint(0,5)]
    for idx in range(prediction[0]):
        prediction.append(
            {
                'translation':f"demo trasnslation of bbox{idx}",
                'point':get_random_polygon()}
            )

    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction



if __name__ == '__main__':
    uvicorn.run('main:app', port=6006, host='0.0.0.0', reload=True,)