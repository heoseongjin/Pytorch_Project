from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    global label_list
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    label_list.append(label)
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    ####
    dog_exist = 0
    cfgfile = "cfg/yolov3.cfg"                      #config파일 선언
    weightsfile = "yolov3.weights"                  #weight파일 선언
    num_classes = 80                                #class개수 정의

    args = arg_parse()                              #argparse를 이용해 명령행을 파싱해오도록 함수 실행
    confidence = float(args.confidence)             #confidence 변수에 --confidence값을 할당
    nms_thesh = float(args.nms_thresh)              #이것도 --nms_thresh값 할당
    start = 0                                       #start는 0
    CUDA = torch.cuda.is_available()                #cuda가 사용가능한 상황인지

    num_classes = 80                                #암튼 80
    bbox_attrs = 5 + num_classes                    #Bouding Box 속성
    
    model = Darknet(cfgfile)                        #Darknet
    model.load_weights(weightsfile)                 #Model에 weighs파일을 load해준다
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()                                #Cuda를 사용중이면 model.cuda()
            
    model.eval()                                    #모델 평가?
    
    videofile = 'video.avi'                         #videofile이름
    
    cap = cv2.VideoCapture(0)                       #videoCapture(0) >> video 캡쳐변수 선언
    
    assert cap.isOpened(), 'Cannot capture source'
    #assert는 가정설정문, 뒤의 조건이 True가 아니면 AssertError를 발생시킨다.
    
    frames = 0
    #frame 변수 선언, 초기값은 0

    start = time.time()                             #시간을 측정해주는 함수
    while cap.isOpened():                           #cap이 초기화가 잘 되어 있는지 확인
        
        ret, frame = cap.read()
        origin_frame = frame

        ####
        frame = cv2.flip(frame, 1)
        #cap.read()는 재생되는 비디오의 한 프레임씩 읽는다.
        #제대로 읽었다면 ret값이 True가 되고, 실패하면 False.
        #읽은 프레임은 frame이다.

        if ret:                                     #ret이 true라면, 제대로 읽었다면
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            #
            
#            im_dim = torch.FloatTensor(dim).repeat(1,2)                        

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim))/inp_dim
            
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            label_list = list()

            list(map(lambda x: write(x, orig_im), output))
            ####
            #if label_list.count('dog') >= 1:
            if label_list.count('remote') >= 1:
                dog_exist = 1
            else:
                dog_exist = 0

            if dog_exist == 1:
                print("Detect--------Detect--------Detect--------Detect--------Detect")
                cv2.imwrite('remote.jpg', origin_frame)

            ####

            cv2.imshow("frame", orig_im)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

        else:
            break
