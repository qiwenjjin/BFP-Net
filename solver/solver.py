#!/usr/bin/env python
# coding=utf-8

import os
import torch
import time
import datetime
import importlib
import shutil
import numpy as np
from solver.basesolver import BaseSolver
from utils.utils import make_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config, divide_amp_pha
from data.data import get_data, get_test_data, get_eval_data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from PIL import Image
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.cfg = cfg
        self.mode = cfg.get('mode', 'train')  # Default to 'train' if mode not specified
        self.nEpochs = cfg.get('nEpochs', 0) if self.mode == 'train' else 1
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1
        self.timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d_%H%M%S')
        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []}

        # GPU/CPU setup
        self.check_gpu()

        # Initialize model
        net_name = cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net
        self.model = net(
            num_channels=cfg['data']['n_colors'],
            channels=64,
            base_filter=64,
            args=cfg
        )
        if self.cuda:
            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)

        # Optimizer, Loss, Scheduler
        if self.mode == 'train':
            self.optimizer = make_optimizer(cfg['schedule']['optimizer'], cfg, self.model.parameters())
            self.loss = make_loss(cfg['schedule']['loss'])
            if cfg['schedule']['use_CosAneal']:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cfg['schedule']['epoch_num'], cfg['schedule']['minimum'])
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg['schedule']['epoch_num'], cfg['schedule']['decay'], last_epoch=-1)
            self.log_name = cfg['dataset'] + '_' + str(cfg['data']['upsacle']) + '_' + str(self.timestamp)
            self.writer = SummaryWriter(cfg['log_dir'] + str(self.log_name))
            save_net_config(self.log_name, self.model)
            save_yml(cfg, os.path.join(cfg['log_dir'] + str(self.log_name), 'config.yml'))
            save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
            save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
            save_config(self.log_name, 'Model parameters: ' + str(sum(param.numel() for param in self.model.parameters())))

        # Load datasets based on mode
        if self.mode == 'train':
            self.train_dataset = get_data(cfg, cfg['data_dir_train'])
            self.train_loader = DataLoader(self.train_dataset, cfg['data']['batch_size'], shuffle=True, num_workers=self.num_workers)
            self.val_dataset = get_data(cfg, cfg['data_dir_eval'])
            self.val_loader = DataLoader(self.val_dataset, cfg['data']['batch_size'], shuffle=False, num_workers=self.num_workers)
        elif self.mode == 'test':
            self.test_dataset = get_test_data(cfg, cfg['test']['data_dir'])
            self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=1, num_workers=self.num_workers)
        elif self.mode == 'eval':
            self.eval_dataset = get_eval_data(cfg, cfg['test']['data_dir'])
            self.eval_loader = DataLoader(self.eval_dataset, shuffle=False, batch_size=1, num_workers=self.num_workers)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            self.num_workers = self.cfg['threads']
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
            gpus_list = self.cfg['gpus']
            self.gpu_ids = [int(str_id) for str_id in gpus_list if int(str_id) >= 0]
            torch.cuda.set_device(self.gpu_ids[0])
            self.loss = self.loss.cuda(self.gpu_ids[0])
        else:
            self.num_workers = 0

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.epoch = ckpt.get('epoch', 1)
            self.records = ckpt.get('records', {'Epoch': [], 'PSNR': [], 'SSIM': [], 'Loss': []})
            self.model.load_state_dict(ckpt['net'])
            if self.mode == 'train':
                self.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            ckpt = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['net'])
            self.epoch = ckpt['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must be less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        if self.mode == 'train':
            self.ckp = {
                'epoch': self.epoch,
                'records': self.records,
                'net': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            save_path = os.path.join(self.checkpoint_dir, self.log_name, 'latest.pth')
            if not os.path.exists(os.path.join(self.checkpoint_dir, self.log_name)):
                os.makedirs(os.path.join(self.checkpoint_dir, self.log_name))
            torch.save(self.ckp, save_path)
            if self.cfg['save_best']:
                if self.records['SSIM'] and self.records['SSIM'][-1] == np.array(self.records['SSIM']).max():
                    shutil.copy(save_path, os.path.join(self.checkpoint_dir, self.log_name, 'bestSSIM.pth'))
                if self.records['PSNR'] and self.records['PSNR'][-1] == np.array(self.records['PSNR']).max():
                    shutil.copy(save_path, os.path.join(self.checkpoint_dir, self.log_name, 'bestPSNR.pth'))

    def train(self):
        with tqdm(total=len(self.train_loader), miniters=1, desc=f'Training Epoch: [{self.epoch}/{self.nEpochs}]') as t:
            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, bms_image, file = batch[0], batch[1], batch[2], batch[3], batch[4]
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])
                self.optimizer.zero_grad()
                self.model.train()
                y, _ = self.model(lms_image, bms_image, pan_image)
                y_amp, y_pha = divide_amp_pha(y)
                if self.cfg['dataset'] == 'QB_remake':
                    ms_image = bms_image
                gnd_amp, gnd_pha = divide_amp_pha(ms_image)
                loss_f = self.loss(torch.log(y_amp + 1e-8), torch.log(gnd_amp + 1e-8)) + self.loss(y_pha, gnd_pha)
                loss = (self.loss(y, ms_image) + self.cfg['schedule']['gamma'] * loss_f) / (self.cfg['data']['batch_size'] * 2)
                epoch_loss += loss.data
                t.set_postfix_str(f"Batch loss {loss.item():.4f}")
                t.update()
                loss.backward()
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['schedule']['gclip'])
                self.optimizer.step()
            self.scheduler.step()
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            save_config(self.log_name, f'Training Epoch {self.epoch}: Loss={self.records["Loss"][-1]:.4f}')
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1, desc=f'Val Epoch: [{self.epoch}/{self.nEpochs}]') as t1:
            psnr_list, ssim_list = [], []
            for iteration, batch in enumerate(self.val_loader, 1):
                ms_image, lms_image, pan_image, bms_image, file = batch[0], batch[1], batch[2], batch[3], batch[4]
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.cuda(self.gpu_ids[0]), lms_image.cuda(self.gpu_ids[0]), pan_image.cuda(self.gpu_ids[0]), bms_image.cuda(self.gpu_ids[0])
                if self.cfg['dataset'] == 'QB_remake':
                    ms_image = bms_image
                self.model.eval()
                with torch.no_grad():
                    y, _ = self.model(lms_image, bms_image, pan_image)
                    loss = self.loss(y, ms_image)
                batch_psnr, batch_ssim = [], []
                y = y[:, 0:3, :, :]
                ms_image = ms_image[:, 0:3, :, :]
                for c in range(y.shape[0]):
                    if not self.cfg['data']['normalize']:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    psnr = calculate_psnr(predict_y, ground_truth, 255)
                    ssim = calculate_ssim(predict_y, ground_truth, 255)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                t1.set_postfix_str(f'Batch loss: {loss.item():.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
                t1.update()
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())
            save_config(self.log_name, f'Val Epoch {self.epoch}: PSNR={self.records["PSNR"][-1]:.4f}, SSIM={self.records["SSIM"][-1]:.6f}')
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)

    def test(self):
        self.model.eval()
        avg_time = []
        for batch in self.test_loader:
            ms_image, lms_image, pan_image, bms_image, name = batch[0], batch[1], batch[2], batch[3], batch[4]
            if self.cuda:
                ms_image = ms_image.cuda(self.gpu_ids[0])
                lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])

            t0 = time.time()
            with torch.no_grad():
                prediction, _ = self.model(lms_image, bms_image, pan_image)
            t1 = time.time()

            if self.cfg['data']['normalize']:
                ms_image = (ms_image + 1) / 2
                lms_image = (lms_image + 1) / 2
                pan_image = (pan_image + 1) / 2
                bms_image = (bms_image + 1) / 2

            print(f"===> Processing: {name[0]} || Timer: {(t1 - t0):.4f} sec.")
            avg_time.append(t1 - t0)
            if self.cfg['dataset'] == 'WV2_8band':
                prediction = prediction.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
                prediction = np.uint8(prediction * 255).astype('uint8')
                for i in range(8):
                    save_dir = f'./saved_data/net_WV2_8/{i+1}'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    imageio.imwrite(f'{save_dir}/{name[0][:-4]}.tif', prediction[:, :, i])
            else:
                self.save_img(bms_image.cpu().data, f"{name[0][:-4]}_bic.tif", mode='CMYK')
                self.save_img(ms_image.cpu().data, f"{name[0][:-4]}_gt.tif", mode='CMYK')
                self.save_img(prediction.cpu().data, f"{name[0][:-4]}.tif", mode='CMYK')
        print(f"===> AVG Timer: {np.mean(avg_time):.4f} sec.")

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
        save_img = np.uint8(save_img * 255).astype('uint8')
        save_dir = os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type']) if self.mode != 'train' else self.checkpoint_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_fn = os.path.join(save_dir, img_name)
        save_img = Image.fromarray(save_img, mode)
        save_img.save(save_fn)

    def run(self):
        if self.mode == 'train':
            if self.cfg['pretrain']['pretrained']:
                self.check_pretrained()
            try:
                while self.epoch <= self.nEpochs:
                    self.train()
                    self.eval()
                    self.save_checkpoint()
                    self.epoch += 1
            except KeyboardInterrupt:
                self.save_checkpoint()
            save_config(self.log_name, 'Training done.')
        elif self.mode == 'test':
            model_path = os.path.join(self.checkpoint_dir, self.cfg['test']['model'])
            self.load_checkpoint(model_path)
            self.test()
        elif self.mode == 'eval':
            model_path = os.path.join(self.checkpoint_dir, self.cfg['test']['model'])
            self.load_checkpoint(model_path)
            self.eval()
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'train', 'test', or 'eval'.")
