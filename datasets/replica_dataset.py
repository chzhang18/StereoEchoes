import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines

import librosa

import torch


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def generate_spectrogram(audioL, audioR, winl=32):
    channel_1_spec = librosa.stft(audioL, n_fft=512, win_length=winl)
    channel_2_spec = librosa.stft(audioR, n_fft=512, win_length=winl)
    spectro_two_channel = np.concatenate((np.expand_dims(np.abs(channel_1_spec), axis=0), np.expand_dims(np.abs(channel_2_spec), axis=0)), axis=0)
    #print(spectro_two_channel.shape)
    return spectro_two_channel


class ReplicaDataset(Dataset):
    def __init__(self, datapath, training):
        self.datadir = datapath
        self.training = training

        #  Replica dataset
        if self.training:
            self.scenes = ['apartment_0', 'apartment_1', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_3',
                           'frl_apartment_4', 'office_0', 'office_1', 'office_2', 'office_3', 'hotel_0', 'room_0', 'room_1', 'room_2']
            #self.scenes = ['apartment_0']
        else:
            self.scenes = ['apartment_2', 'frl_apartment_5', 'office_4']
            #self.scenes = ['office_4']

        self.angle = ['0', '90', '180', '270']

        self.left_fold = 'left/'
        self.right_fold = 'right/'
        self.disp_fold = 'depth/'

        self.audio_datadir = '/home1/zhangchenghao/stereo-echoes/Replica/echoes_navigable/'
        self.audio_type = '3ms_sweep'
        self.audio_shape = [2,257,166]
        self.audio_sampling_rate = 44100
        self.audio_length = 0.06
        self.win_length = 64

        self._read_data()

    def _read_data(self):
        self.left_filenames = []
        self.right_filenames = []
        self.disp_filenames = []
        self.audio_filenames = []

        for scene in self.scenes:
            for angle in self.angle:
                for img in os.listdir(os.path.join(self.datadir, scene, angle, self.left_fold)):
                    left_filename = os.path.join(self.datadir, scene, angle, self.left_fold, img)
                    right_filename = os.path.join(self.datadir, scene, angle, self.right_fold, img)
                    img_num = img.split('.')[0]
                    disp_temp = img_num + '.npy'
                    disp_filename = os.path.join(self.datadir, scene, angle, self.disp_fold, disp_temp)

                    audio_temp = img_num + '.wav'
                    audio_filename = os.path.join(self.audio_datadir, scene, self.audio_type, angle, audio_temp)

                    #print(left_filename, audio_filename)
                    self.left_filenames.append(left_filename)
                    self.right_filenames.append(right_filename)
                    self.disp_filenames.append(disp_filename)
                    self.audio_filenames.append(audio_filename)

    def _depth_to_disp(self, depth, max_disparity=26):
        disparity = depth.copy()
        
        if disparity.min() == 0:
            #import pdb; pdb.set_trace()
            disparity[disparity <= 0] = 0.05
        disparity = 5.0 / disparity # max of depth is 5.0

        return disparity


    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        # left
        left_fname = self.left_filenames[index]
        left_img = np.array(Image.open(left_fname)).astype(np.uint8)

        # right
        right_fname = self.right_filenames[index]
        right_img = np.array(Image.open(right_fname)).astype(np.uint8)

        # depth
        depth_fname = self.disp_filenames[index]
        depth = np.load(depth_fname)
        disparity = self._depth_to_disp(depth)

        # audio
        audio, audio_rate = librosa.load(self.audio_filenames[index], sr=self.audio_sampling_rate, mono=False, duration=self.audio_length)
        #get the spectrogram of both channel
        audio_spec_both = torch.FloatTensor(generate_spectrogram(audio[0,:], audio[1,:], self.win_length))


        if self.training:

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "depth": depth,
                    "disparity": disparity,
                    "audio": audio_spec_both}
        else:

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "depth": depth,
                    "disparity": disparity,
                    "audio": audio_spec_both,
                    "left_filename": left_fname,
                    "top_pad": 0,
                    "right_pad": 0}
