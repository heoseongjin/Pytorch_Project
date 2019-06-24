# Pytorch_YOLO

##### `Pytorch`, `Yolov3`, `Python=3.6`, `Windows10`, `Object_Detection`

## Index

1. 개요
2. **라이브러리** 설치
3. 소스 설명
4. **Yolov3** 다운로드
5. 실행



## 개요

- **Pytorch**와 [**Yolov3**](https://pjreddie.com/media/files/papers/YOLOv3.pdf)를 이용한 **Object_Detection**(객체 검출)

- 사물을 인식한 뒤, 그 중 개(dog)가 인식되면 반응하도록 구현(**존재 유무 판단 중심**)

  ###### [ Detection Example ]

![Detection Example](https://i.imgur.com/m2jwneng.png)



## 라이브러리 설치

- ##### 설치해야할 라이브러리

  - **pytorch**(이미 상위 폴더의 [README](https://github.com/deongjin/Pytorch_Project/blob/master/README.md)에서 설치함)
  - **opencv-python**(OpenCV는 `오픈 소스 컴퓨터 비전 라이브러리`로,  `객체ㆍ얼굴ㆍ행동 인식`, `독순`, `모션 추적` 등의 응용 프로그램에서 사용)
  - **matplotlib**(시각화 라이브러리)
  - **pandas**(Python Data Analysis Library, 데이터 분석 라이브러리)

  

- ##### PIP를 이용한 라이브러리 설치

  - > **pip**는 파이썬으로 작성된 패키지 소프트웨어를 설치 · 관리하는 패키지 관리 시스템이다.

  - Anaconda Prompt를 실행한다![pip_1](./images/pip_1.png)

  - 가상환경 활성화/비활성화

    ```shell
    (base)C:\Users\(username)> conda activate (가상환경 이름)
    ```

    `conda activate`를 통해 설치 되어있는 Pytorch가 설치된 가상환경에 접속한 다.

  - pip install

    ```shell
    (testVenv)C:\Users\(username)> pip install opencv-python matplotlib pandas
    ```

    

## 소스 설명



## Yolov3 다운로드

 [Yolov3 Download](https://pjreddie.com/media/files/yolov3.weights)

## 실행





### On single or multiple images

Clone, and `cd` into the repo directory. The first thing you need to do is to get the weights file
This time around, for v3, authors has supplied a weightsfile only for COCO [here](https://pjreddie.com/media/files/yolov3.weights), and place 

the weights file into your repo directory. Or, you could just type (if you're on Linux)

```
wget https://pjreddie.com/media/files/yolov3.weights 
python detect.py --images imgs --det det 
```

`--images` flag defines the directory to load images from, or a single image file (it will figure it out), and `--det` is the directory
to save images to. Other setting such as batch size (using `--bs` flag) , object threshold confidence can be tweaked with flags that can be looked up with. 

```
python detect.py -h
```

### Speed Accuracy Tradeoff

You can change the resolutions of the input image by the `--reso` flag. The default value is 416. Whatever value you chose, rememeber **it should be a multiple of 32 and greater than 32**. Weird things will happen if you don't. You've been warned. 

```
python detect.py --images imgs --det det --reso 320
```

### On a Camera

Same as video module, but you don't have to specify the video file since feed will be taken from your camera. To be precise, 
feed will be taken from what the OpenCV, recognises as camera 0. The default image resolution is 160 here, though you can change it with `reso` flag.

```
python cam_demo.py
```

You can easily tweak the code to use different weightsfiles, available at [yolo website](https://pjreddie.com/darknet/yolo/)

NOTE: The scales features has been disabled for better refactoring.

### Detection across different scales

YOLO v3 makes detections across different scales, each of which deputise in detecting objects of different sizes depending upon whether they capture coarse features, fine grained features or something between. You can experiment with these scales by the `--scales` flag. 

```
python detect.py --scales 1,3
```

