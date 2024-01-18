from os import listdir
from os.path import join
import random
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import torch
import numpy as np
from os import listdir
import scipy
from scipy import io
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch import nn
import mat73


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", 'tiff', 'bmp', '.mat'])
def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

def load_img(filepath):
    img = Image.open(filepath) #.convert('RGB')
    return img
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
def load_img_gray(filepath):
    img = Image.open(filepath).convert('L')
    return img
class CustomImageDataset2(Dataset):
    def read_data_set(self):
        all_img_files = []
        all_labels = []
        all_img_files2 = []

        class_names = os.walk(self.data_set_path).__next__()[1]
        class_names2 = os.walk(self.path).__next__()[1]

        for index, class_name in enumerate(class_names):   ##nir_3ch
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                mat = mat73.loadmat(img_file)
                # mat = Image.open(img_file)
                if mat is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        for index, class_name2 in enumerate(class_names2):   ## nir image
            img_dir = os.path.join(self.path, class_name2)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files2.append(img_file)

        print(all_img_files2)


        return all_img_files, all_labels, len(all_img_files), len(class_names), all_img_files2

    def __init__(self, data_set_path, path, transforms=None):
        self.data_set_path = data_set_path    ###nir_56ch mat 파일 경로
        self.path = path        ### nir_1ch img 파일 경로

        self.image_files_path, self.labels, self.length, self.num_classes, self.all_img_files2= self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):

        nir_img = load_img_gray(self.all_img_files2[index])  ## rgb image 를 gray image로 받는다.
        nir_img = np.asarray(nir_img)

        h,w = nir_img.shape
        nir_img = torch.from_numpy(nir_img)
        nir_img = torch.unsqueeze(nir_img,dim=0)
        # print(nir_img.shape)
        h_ = np.random.randint(0,h-256)
        w_ = np.random.randint(0,w-256)
        mat_56ch = mat73.loadmat(self.image_files_path[index])
        mat_56ch = mat_56ch['nir_hs']
        mat_56ch = np.asarray(mat_56ch)

        labels = self.labels[index]

        mat_56ch = torch.from_numpy(mat_56ch)
        nir_img = nir_img[:,h_:h_+256,w_:w_+256]
        nir_img = nir_img.float()
        nir_img = nir_img/255


        mat_56ch = mat_56ch[h_:h_+256,w_:w_+256,:]
        mat_56ch = mat_56ch.float()
        mat_56ch = mat_56ch/255.
        mat_56ch.transpose_(0,2).transpose_(1,2)
        final = torch.cat([nir_img],dim=0)


        return final.float(), labels

    def __len__(self):
        return self.length
class CustomImageDataset2_test(Dataset):
    def read_data_set(self):
        all_img_files = []
        all_labels = []
        all_img_files2 = []

        class_names = os.walk(self.data_set_path).__next__()[1]
        class_names2 = os.walk(self.path).__next__()[1]

        for index, class_name in enumerate(class_names):   ##nir_3ch
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                mat = mat73.loadmat(img_file)
                # mat = Image.open(img_file)
                if mat is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        for index, class_name2 in enumerate(class_names2):   ## nir image
            img_dir = os.path.join(self.path, class_name2)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files2.append(img_file)

        print(all_img_files2)


        return all_img_files, all_labels, len(all_img_files), len(class_names), all_img_files2

    def __init__(self, data_set_path, path, transforms=None):
        self.data_set_path = data_set_path    ###nir_56ch mat 파일 경로
        self.path = path        ### nir_1ch img 파일 경로

        self.image_files_path, self.labels, self.length, self.num_classes, self.all_img_files2= self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        # nir_1ch = load_img(self.all_img_files2[index])  ## nir_1ch image 불러오기
        # nir_1ch = np.asarray(nir_1ch)
        nir_img = load_img_gray(self.all_img_files2[index])  ## rgb image 를 gray image로 받는다.
        nir_img = np.asarray(nir_img)
        nir_img = torch.from_numpy(nir_img)
        nir_img = torch.unsqueeze(nir_img,dim=0)
        nir_img = nir_img.float()
        nir_img = nir_img/255.

        mat_56ch = mat73.loadmat(self.image_files_path[index])
        mat_56ch = mat_56ch['nir_hs']
        mat_56ch = np.asarray(mat_56ch)

        labels = self.labels[index]

        mat_56ch = torch.from_numpy(mat_56ch)

        mat_56ch = mat_56ch.float()
        mat_56ch = mat_56ch/255.
        mat_56ch.transpose_(0,2).transpose_(1,2)


        final = torch.cat([nir_img],dim=0)


        return final.float(), labels

    def __len__(self):
        return self.length
class CustomImageDataset3(Dataset):
    def read_data_set(self):
        all_img_files = []
        all_labels = []
        all_img_files2 = []
        all_img_files3 = []
        all_img_files4 = []
        all_img_files5 = []
        class_names = os.walk(self.data_set_path).__next__()[1]
        class_names2 = os.walk(self.path).__next__()[1]
        class_names3 = os.walk(self.path2).__next__()[1]
        class_names4 = os.walk(self.path3).__next__()[1]
        class_names5 = os.walk(self.path4).__next__()[1]
        for index, class_name in enumerate(class_names):   ##nir_3ch
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                mat = scipy.io.loadmat(img_file)
                # mat = Image.open(img_file)
                if mat is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)
        for index, class_name2 in enumerate(class_names2):   ## nir image
            img_dir = os.path.join(self.path, class_name2)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files2.append(img_file)
        for index, class_name3 in enumerate(class_names3):   ## rgb image
            img_dir = os.path.join(self.path2, class_name3)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files3.append(img_file)
        for index, class_name4 in enumerate(class_names4):   ## nir_8ch
            img_dir = os.path.join(self.path3, class_name4)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                mat = scipy.io.loadmat(img_file)
                # img = Image.open(img_file)
                if mat is not None:
                    all_img_files4.append(img_file)
        for index, class_name5 in enumerate(class_names5):   ## nir_5ch
            img_dir = os.path.join(self.path4, class_name5)
            img_files = os.walk(img_dir).__next__()[2]
            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                mat = scipy.io.loadmat(img_file)
                # img = Image.open(img_file)
                if mat is not None:
                    all_img_files5.append(img_file)
        # print(all_img_files4)
        print(all_img_files2)
        print(all_img_files5)

        return all_img_files, all_labels, len(all_img_files), len(class_names), all_img_files2, all_img_files3, all_img_files4, all_img_files5

    def __init__(self, data_set_path, path, path2, path3,path4,transforms=None):
        self.data_set_path = data_set_path    ###nir_3ch mat 파일 경로
        self.path = path        ### nir_1ch img 파일 경로
        self.path2 = path2       ### rgb 파일 경로
        self.path3 = path3    ## nir_8ch 파일 경로
        self.path4 = path4   ## nir_5ch 파일 경로
        self.image_files_path, self.labels, self.length, self.num_classes, self.all_img_files2, self.all_img_files3 , self.all_img_files4 , self.all_img_files5= self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        # nir_1ch = load_img(self.all_img_files2[index])  ## nir_1ch image 불러오기
        # nir_1ch = np.asarray(nir_1ch)
        rgb_img = load_img(self.all_img_files3[index])  ## rgb image 를 gray image로 받는다.
        rgb_img = np.asarray(rgb_img)
        h,w,_ = rgb_img.shape
        # gray_img = load_img_gray(self.all_img_files3[index])
        # gray_img = np.asarray(gray_img)
        # mat_8ch = scipy.io.loadmat(self.all_img_files4[index])
        # mat_8ch = mat_8ch['mat_data']
        # mat_8ch = np.asarray(mat_8ch)
        # mat_5ch = scipy.io.loadmat(self.all_img_files5[index])
        # print(self.all_img_files5[index])
        # mat_5ch = mat_5ch['mat_data']
        # mat_5ch = np.asarray(mat_5ch)
        # h, w ,_= mat_8ch.shape
        h_ = np.random.randint(0,h-256)
        w_ = np.random.randint(0,w-256)
        mat_3ch = scipy.io.loadmat(self.image_files_path[index])
        mat_3ch = mat_3ch['mat_data']
        mat_3ch = np.asarray(mat_3ch)

        labels = self.labels[index]
        # if self.transforms is not None:
            # rgb_img = self.transforms(rgb_img)
            # nir_1ch = self.transforms(nir_1ch)
            # gray_img = self.transforms(gray_img)
            # mat_3ch_1 = self.transforms(mat_3ch_1)
            # mat_3ch_2 = self.transforms(mat_3ch_2)
            # mat_3ch_3 = self.transforms(mat_3ch_3)
            # mat_8ch_1 = self.transforms(mat_8ch_1)
            # mat_8ch_2 = self.transforms(mat_8ch_2)
            # mat_8ch_3 = self.transforms(mat_8ch_3)
            # mat_8ch_4 = self.transforms(mat_8ch_4)
            # mat_8ch_5 = self.transforms(mat_8ch_5)
            # mat_8ch_6 = self.transforms(mat_8ch_6)
            # mat_8ch_7 = self.transforms(mat_8ch_7)
            # mat_8ch_8 = self.transforms(mat_8ch_8)
            # mat_8ch = self.transforms(mat_8ch)
            # mat_3ch = self.transforms(mat_3ch)



        # rgb_img = torch.from_numpy(rgb_img)
        # nir_1ch = torch.from_numpy(nir_1ch)
        mat_3ch = torch.from_numpy(mat_3ch)
        # mat_3ch = torch.cat([mat_3ch_1,mat_3ch_2,mat_3ch_3],dim=0)
        # mat_8ch = torch.cat([mat_8ch_1,mat_8ch_2,mat_8ch_3,mat_8ch_4,mat_8ch_5,mat_8ch_6,mat_8ch_7,mat_8ch_8],dim=0)
        # rgb_img = torch.from_numpy(rgb_img)
        # rgb_img = rgb_img[h_:h_+256,w_:w_+256,:]
        # rgb_img = rgb_img.float()
        # rgb_img = rgb_img/255.
        # gray_img = torch.from_numpy(gray_img)
        # gray_img = gray_img[h_:h_+256,w_:w_+256]
        # gray_img = torch.unsqueeze(gray_img,dim=2)
        # gray_img = gray_img.float()
        # gray_img = gray_img/255.
        # nir_1ch = nir_1ch[h_:h_+256,w_:w_+256,:]
        # nir_1ch = nir_1ch.float()
        # nir_1ch = nir_1ch/255.


        mat_3ch = mat_3ch[h//2-128:h//2+128,w//2-128:w//2+128,:]
        mat_3ch = mat_3ch.float()
        mat_3ch = mat_3ch/255.
        # mat_8ch = torch.from_numpy(mat_8ch)
        # mat_8ch = mat_8ch[h_:h_ + 256, w_:w_ + 256,:]
        # mat_8ch = mat_8ch.float()
        # mat_8ch = mat_8ch / 255.
        # mat_5ch = torch.from_numpy(mat_5ch)
        # mat_5ch = mat_5ch[h_:h_+256,w_:w_+256,:]
        # mat_5ch = mat_5ch.float()
        # mat_5ch = mat_5ch/255.

        # gray_img.transpose_(0,2).transpose(1,2)
        # rgb_img.transpose_(0,2).transpose(1,2)
        # nir_1ch.transpose_(0,2).transpose_(1,2)
        mat_3ch.transpose_(0,2).transpose_(1,2)
        # mat_8ch.transpose_(0, 2).transpose_(1, 2)  ###[8 256 256]
        # mat_5ch.transpose_(0,2).transpose_(1,2)

        # nir_1ch = np.asarray(nir_1ch)
        # nir_1ch = nir_1ch[:,h_:h_+256,w_:w_+256]
        # nir_1ch = nir_1ch / 255.
        # nir_1ch = torch.from_numpy(nir_1ch)
        # nir_1ch.transpose_(1,2)

        final = torch.cat([mat_3ch],dim=0)
        # print(final.shape)





        # os.system('pause')




        return final.float(), labels

    def __len__(self):
        return self.length