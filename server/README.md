# OCR In Editor's Server via FastAPI
해당 프로젝트는 [부스트캠프 최종프로젝트](https://github.com/boostcampaitech2/final-project-level3-cv-07)를 위한 BackEnd 개발에 관한 내용입니다.

## 기능
1. 이미지 POST 
    - FastAPI를 사용해서 API 생성
    - 이미지를 받고 번역된 문장, Bounding Box, Word&Backgroud 색상을 Responese
    - [Swagger](http://www.realbro.site/docs)를 통해 API문서 관리
2. OCR Model Inference
    - [모델 팀](https://github.com/boostcampaitech2/final-project-level3-cv-07/tree/main/models)에서 연구 & 개발한 모델을 사용
    - API를 통해 받은 이미지를 Inference하여 Image내의 Text와 Bounding Box정보를 얻음
3. Get background & word color
    - Bounding Box내의 Image Histogram 정보를 사용, Background와 Word의 색상을 추출
    - API를 통해 받은 이미지를 Inference하여 Image내의 Text와 Bounding Box정보를 얻음
    - 기존 K-means를 사용하여 이미지의 색상을 추출하는 방법을 생각 했으나 \
    Response Time이 10~20초이상 걸리는 문제점이 생겨 Histgram 방식으로 변경, 5초 이내의 Response 가능
4. Text Translate with Papago
    - 최종 번역된 결과를 얻기 위해 모델에서 Inference한 Text를 Papago API를 통해 번역 향후 이 또한 번역 모델로 사용 가능할것으로 예상
5. CI & CD
    - 효율적인 CI & CD를 위해 Github Action과 Docker사용
    - 이를 위해 개발 서버와 운용 서버의 분리 후 운용 서버 [도메인](http://realbro.site/docs)
    - Poetry를 사용한 의존성 관리
    - Main Brunch에 merge or push되면 자동으로 ssh사용 서버 접속, pull한 후 Dockerize하여 배포
    - 처음 Docker Image Build시에 필요한 모든 것들을 새로 설치하여 Image를 재생성하는 과정 때문에 10분 넘게 걸리는 문제점 존재 \
    이후 Base Docker Image 생성하여 재배포 시간 30초 이내 시행
    - Github Action을 통한 CI & CD의 결과를 즉각적으로 알 수 있도록 Slack으로 결과 전송

## Prerequisites


```
cd models

pip install -r requirements.txt
```

## Dataset Structure

During training, we use [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads). In addition, we use [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) Test Set for validating our model.<br /><br />

The dataset should structure as follows:

```
[dataset root directory]
├── train_images
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── train_gts
│   ├── gt_img_1.txt
│   ├── gt_img_2.txt
│   └── ...
└── test_images
    ├── img_1.jpg
    ├── img_2.jpg
    └── ...
```
Note: the [dataset root directory] should be placed in "config.json" file. <br /><br />
Sample of ground truth format:
```
x1,y1,x2,y2,x3,y3,x4,y4,transcription
```


## Training

Training the model by yourself
```
python train.py
```
Note: check the "config.json" file, which is used to adjust the training configuration.<br /><br />

## Inference
Test the model you trained
```
python eval.py
```

If you want to change "your model, input images, output folder", check the eval.py file.

## Running server

Running the demo of our pre-trained [model](https://drive.google.com/file/d/1toEqT1LA-0ieY0ZFeKc6UWJOVvPXDtF1/view?usp=sharing)

```
python demo.py -m=model_best.pth.tar
```

## Acknowledgments

* https://github.com/ishin-pie/e2e-scene-text-spotting
* https://github.com/argman/EAST
* https://github.com/jiangxiluning/FOTS.PyTorch

