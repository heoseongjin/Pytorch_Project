import os
import librosa
import librosa.display
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def main(args):
    urbansound_folder = args.urbansound_dir
    urbansound_graph_folder = urbansound_folder + os.sep + 'graph'
    urbansound_graph_mfcc_folder = urbansound_folder + os.sep + 'graph_mfcc'
    urbansound_dogbark_graph_folder = urbansound_graph_folder + os.sep + 'positive'
    urbansound_other_graph_folder = urbansound_graph_folder + os.sep + 'negative'
    urbansound_dogbark_graph_mfcc_folder = urbansound_graph_mfcc_folder + os.sep + 'positive'
    urbansound_other_graph_mfcc_folder = urbansound_graph_mfcc_folder + os.sep + 'negative'

    if not os.path.exists(urbansound_graph_mfcc_folder):
        os.mkdir(urbansound_graph_mfcc_folder)
    if not os.path.exists(urbansound_dogbark_graph_mfcc_folder):
        os.mkdir(urbansound_dogbark_graph_mfcc_folder)
    if not os.path.exists(urbansound_other_graph_mfcc_folder):
        os.mkdir(urbansound_other_graph_mfcc_folder)

    '''
    for file in os.listdir(urbansound_dogbark_graph_folder):
        filename, extension = os.path.splitext(file)
        
        if extension == '.wav':
            # open sound file
            audiopath = urbansound_dogbark_graph_folder + os.sep + file
            print(audiopath)

            y, sr = librosa.load(audiopath)
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

            plt.savefig(urbansound_dogbark_graph_mfcc_folder + '/' + filename + '.png')
            plt.close(fig)
    '''
    '''
    for file in os.listdir(urbansound_other_graph_folder):
        filename, extension = os.path.splitext(file)
        if extension == '.wav':
            # open sound file
            audiopath = urbansound_other_graph_folder + os.sep + file
            print(audiopath)

            y, sr = librosa.load(audiopath)
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

            plt.savefig(urbansound_other_graph_mfcc_folder + '/' + filename + '.png')
            plt.close(fig)
    '''

    index = 0

    for file in os.listdir(urbansound_other_graph_folder):
        filename, extension = os.path.splitext(file)
        index += 1

        if extension == '.wav' and index % 200 == 0:
        #if extension == '.wav':
            # open sound file
            audiopath = urbansound_other_graph_folder + os.sep + file
            print(audiopath)

            y, sr = librosa.load(audiopath)
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

            plt.savefig(urbansound_other_graph_mfcc_folder + '/' + filename + '.png')
            plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--urbansound_dir', '-u', dest='urbansound_dir', required=True)
    args = parser.parse_args()
    main(args)

