# adapted from https://github.com/qingsenyangit/AHDRNet with revision
import os
import random
import numpy as np
import torch
import h5py
import time
import glob

import torch.nn as nn
from torch.nn import init
import torchvision as tv
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from glob import glob

import cv2
import imageio
from utils import AverageLoss
from torch.utils.tensorboard import SummaryWriter

imageio.plugins.freeimage.download()
def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def model_restore(model, trained_model_dir):
    model_list = glob.glob((trained_model_dir + "/*.pkl"))
    a = []
    for i in range(len(model_list)):
        index = int(model_list[i].split('model')[-1].split('.')[0])
        a.append(index)
    epoch = np.sort(a)[-1]
    model_path = trained_model_dir + 'trained_model{}.pkl'.format(epoch)
    model.load_state_dict(torch.load(model_path))
    return model, epoch

def model_restore_test(model, trained_model_dir, trained_model_filename):
    model_path = os.path.join(trained_model_dir, trained_model_filename)
    model.load_state_dict(torch.load(model_path))
    return model



class data_loader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

        data = []
        label=[]
        for i in range(self.length):
            sample_path = self.list_txt[i][:-1]
            f = h5py.File(sample_path, 'r')

            data.append(f['IN'][:])
            label.append(f['GT'][:])
            f.close()



    def __getitem__(self, index):

        sample_path = self.list_txt[index][:-1]

        if os.path.exists(sample_path):

            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
            crop_size = 256
            data, label = self.imageCrop(data, label, crop_size)
            data, label = self.image_Geometry_Aug(data, label)


        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

    def random_number(self, num):
        return random.randint(1, num)

    def imageCrop(self, data, label, crop_size):
        c, w, h = data.shape
        w_boder = w - crop_size  # sample point y
        h_boder = h - crop_size  # sample point x ...

        start_w = self.random_number(w_boder - 1)
        start_h = self.random_number(h_boder - 1)

        crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
        return crop_data, crop_label

    def image_Geometry_Aug(self, data, label):
        c, w, h = data.shape
        num = self.random_number(4)

        if num == 1:
            in_data = data
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data = data[:, :, index]
            in_label = label[:, :, index]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data = data[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data = in_data[:, :, index]
            in_label = in_label[:, :, index]

        return in_data, in_label


class testimage_dataloader(data.Dataset):
    def __init__(self, list_dir):
        f = open(list_dir)
        self.list_txt = f.readlines()
        self.length = len(self.list_txt)

        data = []
        label=[]
        for i in range(self.length):
            sample_path = self.list_txt[i][:-1]
            f = h5py.File(sample_path, 'r')

            data.append(f['IN'][:])
            label.append(f['GT'][:])
            f.close()



    def __getitem__(self, index):

        sample_path = self.list_txt[index][:-1]

        if os.path.exists(sample_path):

            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

class testimage_dataloader_tursun(data.Dataset):
    def __init__(self, list_dir='./tursun_dataset/dataset/*'):
        self.image_pair_dir_list = glob(list_dir)

        self.length = len(self.image_pair_dir_list)

        data = []
        label=[]
        for i in range(self.length):
            sample_path = self.list_txt[i][:-1]
            f = h5py.File(sample_path, 'r')

            data.append(f['IN'][:])
            label.append(f['GT'][:])
            f.close()



    def __getitem__(self, index):

        sample_path = self.list_txt[index][:-1]

        if os.path.exists(sample_path):

            f = h5py.File(sample_path, 'r')
            data = f['IN'][:]
            label = f['GT'][:]
            f.close()
        # print(sample_path)
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.length

def get_lr(epoch, lr, max_epochs):
    if epoch <= max_epochs * 0.75:
        lr = lr
    else:
        lr = 0.1 * lr
    return lr

def train(epoch, model, train_loaders, optimizer, averageLoss, args):
    # tensorboard writer
    
    lr = get_lr(epoch, args.lr, args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    model.train()
    num = 0
    trainloss = 0
    start = time.time()
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)
    for batch_idx, (data, target) in enumerate(train_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        end = time.time()
        
############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)

        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)

        #writer.add_graph(model=model, input_to_model=[data1, data2, data3])
        optimizer.zero_grad()
        output = model(data1, data2, data3)
            
        #########  make the loss
        output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        trainloss = F.l1_loss(output, target)

        trainloss.backward()
        optimizer.step()
        
        
        averageLoss.update(trainloss)
        if (batch_idx +1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx, trainloss.data))
            fname = args.trained_model_dir + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss = 0

def test_without_label(model, args):
    dataset_path = './tursun_dataset/dataset/*'
    image_pair_dir_list = glob(dataset_path)
    output_path = './tursun_dataset/result/'
    i = 0
    print(sorted(image_pair_dir_list))
    for image_pair_dir in sorted(image_pair_dir_list):
        image1 = os.path.join(image_pair_dir, '5.tiff')
        image2 = os.path.join(image_pair_dir, '6.tiff')
        image3 = os.path.join(image_pair_dir, '7.tiff')
        image1 = imageio.imread(image1) / (2**16)
        image2 = imageio.imread(image2) / (2**16)
        image3 = imageio.imread(image3) / (2**16)
        
        print(np.max(image1))
        image1 = torch.from_numpy(image1).transpose(0, 2).float()
        image2 = torch.from_numpy(image2).transpose(0, 2).float()
        image3 = torch.from_numpy(image3).transpose(0, 2).float()
        image1_HDR = torch.as_tensor(LDR_to_HDR(image1, 0))
        image2_HDR = torch.as_tensor(LDR_to_HDR(image2, 2))
        image3_HDR = torch.as_tensor(LDR_to_HDR(image3, 4))
        data1 = Variable(torch.cat((image1, image1_HDR), 0))
        data2 = Variable(torch.cat((image2, image2_HDR), 0))
        data3 = Variable(torch.cat((image3, image3_HDR), 0))
        if args.use_cuda:
            data1 = data1.cuda()
            data2 = data2.cuda()
            data3 = data3.cuda()
        model.eval()
        with torch.no_grad():
            output = model(data1.unsqueeze(0), data2.unsqueeze(0), data3.unsqueeze(0))
        output = torch.squeeze(output, 0)
        print(torch.max(output))
        output = output.data.cpu().numpy().astype(np.float32).transpose(2, 1, 0)
    
        imageio.imwrite('./{}/{}_our.hdr'.format(output_path, i), output, format='HDR-FI')
        mu_pred = tone_mapping(output)
        cv2.imwrite('./{}/{}_our.png'.format(output_path, i), cv2.cvtColor(255*mu_pred, cv2.COLOR_RGB2BGR))
        i+=1

def testing_fun(model, test_loaders, args):
    model.eval()
    total_time = []
    for batch_idx, (data, target) in enumerate(test_loaders):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        

############  used for End-to-End code
        data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)
        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        target = Variable(target)

        with torch.no_grad():
            start = time.time()
            output = model(data1, data2, data3)
            end = time.time()
            time_ = end - start
            total_time.append(time_)
        output = torch.squeeze(output.clone(), 0)
        target = torch.squeeze(target.clone(), 0)
        output = output.data.cpu().numpy().astype(np.float32).transpose(2, 1, 0)
        target = target.data.cpu().numpy().astype(np.float32).transpose(2, 1, 0)
        mu_pred = tone_mapping(output)
        mu_label = tone_mapping(target)
        # save the tone mapped image
        cv2.imwrite('./result_tone_map/{}_ourmodel.png'.format(batch_idx), cv2.cvtColor(255*mu_pred, cv2.COLOR_RGB2BGR))
        cv2.imwrite('./result_tone_map/{}_label.png'.format(batch_idx), cv2.cvtColor(255*mu_label, cv2.COLOR_RGB2BGR))
        # save the hdr image
        imageio.imwrite('./result_hdr/{}_ourmodel.hdr'.format(batch_idx), output, format='HDR-FI')
        imageio.imwrite('./result_hdr/{}_label.hdr'.format(batch_idx), target, format='HDR-FI')

def tone_mapping(x):
	mu = 5000
	return np.log(1+mu*x)/np.log(1+mu)

def LDR_to_HDR(image, exposure, gamma=2.2):
    return np.power(image,gamma)/(2**exposure)