#Copyright (c) 2022-2023 Lingkai Zhu, Uppsala University, Sweden

import os
import torch
from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt

def mk_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def model_load(model, trained_model_dir, model_file_name):
    model_path = os.path.join(trained_model_dir, model_file_name)
    model.load_state_dict(torch.load(model_path))
    return model

class AverageLoss(Metric):
    full_state_update: bool = False
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
    def update(self, total_loss):
        self.steps += 1
        self.loss += total_loss

    def compute(self):
        return self.loss / self.steps

def draw_training(epoch, loss_sup):
    t = np.arange(1, epoch + 1, 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(t, loss_sup, color='green', label='loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('PSNR-mu', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, psnr_mu, color=color, label='PSNR-mu')
    # # ax2.plot(t, dice_val, color='brown', label='dice_val')
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./train_curve.png")
    plt.close()

def draw_training_with_test(epoch, loss_sup, psnr_mu):
    t = np.arange(1, epoch + 1, 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(t, loss_sup, color='green', label='loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('PSNR-mu', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, psnr_mu, color=color, label='PSNR-mu')
    # ax2.plot(t, dice_val, color='brown', label='dice_val')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("./train_curve.png")
    plt.close()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)