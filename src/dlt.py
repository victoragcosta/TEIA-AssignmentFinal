"""Dataset Load and Tread

Load data from ../data/genres/* get features (spectrogram, chroma stft, ...) and save in CSV file
The class DLT have de following functions:
    - __init__ : Initialize object defining some options and others like number of audios of each class
    - get_train: Reads audio files and extracts features, and return/save all in CSV file, until get the percent setted for train
    - get_test: Reads audio files and extract features, and return/save all in CSV file, until get the 1-percent setted for train
    - __read_one_audio: Read one audio, remove samples of it, and split, getting features for each and returning

And the module have one function:
    - plot_data: Plot the given features

e.g.:
    dlt_obj = DLT()
    data = dlt_obj.get_train(n_audio=10)
    for x in range(10):
        dlt.plot_data(data[x][1],
                      title=data[3*x][0],
                      save=True,
                      imagem_path='..',
                      image_name='spectrogram_' + data[3*x][0])
"""

import random as rnd
import librosa as lb
import librosa.display

import matplotlib.pyplot as plt
import numpy as np

DEBUG = False


class DLT:

    def __init__(self, shuffler=False, format='spectrogram', train_percent=70, sample_t_size=10,
                 sample_split=5):
        """Initialize a Dataset load


        :param shuffler:(boolean, default=False) - If True at call of 'get_train' / 'get_test' the returned values will be shuffled
        :param format:(string['spectrogram','chroma_stft'], default='spectrogram) -  Define the feature that will be extracted in gets functions
        :param train_percent:(int, default=70) - Percent of total data that will be getted from 'get_train' function
        :param sample_t_size:(int or None, default=10) - Window size of sample extracted in each audio in seconds. If None or 0 is setted, sample_t_size will be seted to maximum size (audio size). If audio is bigger than window size, the gets functions will return many samples of the same music, as many as possible. Be carefull with large values, if bigger than audio causes error. All audios in default DB have 30s
        :param sample_split:(int, default=5) - Split the sample extract in audio. Used to train many models at same time and get majority vote
        """
        self.sample_split = sample_split
        self.sample_t_size = sample_t_size
        self.shuffler = shuffler
        self.labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.n_unread_audios = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        self.n_audios_train = (sum(self.n_unread_audios) * train_percent) // 100
        self.n_audios_test = (sum(self.n_unread_audios) * (100 - train_percent)) // 100
        self.path='data/genres'

        if format == 'spectrogram':
            self.f_function = lambda x: librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
        elif format == 'chroma_stft':
            self.f_function = lambda x: librosa.feature.chroma_stft(x)
        else:
            raise Exception("Invalid Format!")

    def get_train(self, n_audio=100, save_CSV=False, file_path="", file_name=""):
        """Return/save the set of train based in train_percent attribute

        :param n_audio:(int, default=100) - number of audios to read. PS: the quantity of samples will be n_audio*(audio_duration//sample_t_size)
        :param save_CSV:(boolean, default=False) - If True save in CSV file the returned values
        :param file_path:(string, default="") - Path to save file
        :param file_name:(string, default="") - Name of file to save

        :return:(array-like) Return all extracted samples and the splits of each sample, with label number of each sample.
        format: n_sample * (label, n_splits*[feature_shape])
        """

        if self.n_audios_train <= 0:
            return None
        elif n_audio > self.n_audios_train:
            n_audio = self.n_audios_train

        self.n_audios_train -= n_audio

        ret = []

        for i in range(n_audio):
            genre_index = self.n_unread_audios.index(max(self.n_unread_audios))
            ret = ret + self.__read_one_audio(genre_index)

        return rnd.shuffle(ret) if self.shuffler else ret

    def get_test(self, n_audio=100):
        """Return/save the set of train based in train_percent attribute

        :param n_audio:(int, default=100) - number of audios to read. PS: the quantity of samples will be n_audio*(audio_duration//sample_t_size)
        :return:(array-like) - Return all extracted samples and the splits of each sample, with label number of each sample.
        format: n_sample * (label, n_splits*[feature_shape])
        """

        if self.n_audios_test <= 0:
            return None
        elif n_audio > self.n_audios_test:
            n_audio = self.n_audios_test

        self.n_audios_test -= n_audio

        ret = []

        for i in range(n_audio):
            genre_index = self.n_unread_audios.index(max(self.n_unread_audios))
            ret = ret + self.__read_one_audio(genre_index)

        return rnd.shuffle(ret) if self.shuffler else ret

    def get_labels(self):
        """Return all labels name in order

        :return: (array-like) - All classes name
        format: n_classes * [string]
        """
        return self.labels

    def __read_one_audio(self, genre_index):
        """Read, extract samples, split and gets features of one audio

        :param genre_index:(int) - index of labels array to be read
        :return: (array-like) - Return all extracted samples and the splits of each sample, with label number of each sample, for one audio file.
        format: (audio_duration // sample_t_size) * (label, n_splits*[feature_shape])
        """
        audio_name = "{path}/{folder}/{genre}.{n:05}.au".format(
            path=self.path,
            folder=self.labels[genre_index],
            genre=self.labels[genre_index],
            n=100-self.n_unread_audios[genre_index])

        audio = lb.core.load(audio_name)[0]
        audio_duration = lb.core.get_duration(audio)
        audio_n_samples = (np.math.floor(audio_duration)//self.sample_t_size) if self.sample_t_size else 1
        audio_sample_size = len(audio) // audio_n_samples

        ret = []

        for i in range(audio_n_samples):
            partial_ret = []
            sample_start = audio_sample_size * i
            sample_end = (audio_sample_size * (i + 1)) - 1

            for j in range(self.sample_split):
                split_start = (audio_sample_size//self.sample_split)*j
                split_end = (audio_sample_size//self.sample_split)*(j+1)

                audio_cut = audio[(sample_start + split_start):(sample_end + split_end)]

                partial_ret.append(
                    self.f_function(audio_cut)
                )

                plt.show()

            ret.append([genre_index, partial_ret])

        if DEBUG:
            print("read from: {path} -> {n_splits} splits; {n_samples} samples".format(path=audio_name,
                                                                                       n_splits= self.sample_split,
                                                                                       n_samples=audio_n_samples))
        self.n_unread_audios[genre_index] -= 1

        return ret


def plot_data(data, x_axis='time', y_axis='log', title='', plot=True, save=False, image_path="", image_name=""):
    """Plot/save features of audios in one image

    :param data:(array-like) - Array of features of audios, e.g. array of spectrogram
    :param x_axis: (None or string, default: 'time') - axis x following the options described in https://librosa.github.io/librosa/generated/librosa.display.specshow.html
    :param y_axis: (None or string, default: 'log') - axis y options
    :param title: (string, default: '') - title of plot
    :param plot: (boolean, default: True) - If True shot plotted image
    :param save: (boolean, default: False) - If True save in file the plotted image
    :param image_path: (string, default: '') - Image path to be saved
    :param image_name: (string, default: '') - Image name to be saved
    """
    lines = (1 + len(data) // 2)
    columns = 2

    for i, d in enumerate(data):
        plt.subplot(lines, columns, (i+1))
        lb.display.specshow(d, x_axis=x_axis, y_axis=y_axis)
        plt.colorbar()

    plt.suptitle(title)

    if save:
        plt.savefig(image_path + '/' + image_name)
    if plot:
        plt.show()
