import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from test import CustomConvNet, CustomImageDataset, custom_test

import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "mic.wav"

######hyper_param_epoch = 20
hyper_param_epoch = 20
hyper_param_batch = 8
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomRotation(10.), transforms.ToTensor()])
transforms_test = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="./data/train", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)
test_data_set = CustomImageDataset(data_set_path="./data/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)
custom_test_data_set = CustomImageDataset(data_set_path="./data/custom", transforms=transforms_test)
custom_test_loader = DataLoader(custom_test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)

# custom_model.load_state_dict(torch.load("./model/test_model.pth"))
custom_model.load_state_dict(torch.load('./model/test_model.pth', map_location=lambda storage, loc: storage))

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


def record():
    #print("* recording")
    frames = []
    for iii in range(0, RATE // (CHUNK * RECORD_SECONDS * 2)):          #0.5초로 바꿔봄
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    #print("* done recording")

    #stream.close()
    #pa.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def detection(num):
    print(str(num)+' - record')
    record()

    y, sr = librosa.load('mic.wav')
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    fig = plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    plt.title('mel power spectrogram')
    # plt.colorbar(format = '%+02.0f db')
    plt.tight_layout()
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('./data/custom/dog/mic.png')

    custom_test()

    # plt.show()
    plt.close(fig)


for i in range(0, 100):
    start = time.time()
    detection(i)
    runtime = time.time()-start
    print('Time: %.3f[sec]' %runtime)

stream.close()
