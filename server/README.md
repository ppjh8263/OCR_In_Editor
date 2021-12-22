# OCR In Editor's Server via FastAPI
해당 프로젝트는 [부스트캠프 최종프로젝트](https://github.com/boostcampaitech2/final-project-level3-cv-07)를 위한 BackEnd 개발에 관한 내용입니다.

## 기능
1. 이미지 POST 
    - FastAPI를 사용해서 API 생성
    - 이미지를 효율적으로 주고 받기 위해서 base64 iamge로 디코딩하여 사용
    - [Swagger](http://www.realbro.site/docs)를 통해 API문서 관리
2. OCR Model Inference
3. Get background & word color
4. Text Translate with Papago

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

