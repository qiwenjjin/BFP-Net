clear all
clc
close all

N = 1;
value_mat = zeros([N * 2,10]);
% Refference
% pan_path = '/home/pcl/BaiXu/MyNet/FusionNN/GF-2/Contrast/GPPNN20223172132/TestFolder/pan/'
% pan_label_path ='/home/pcl/BaiXu/MyNet/FusionNN/GF-2/Contrast/GPPNN20223172132/TestFolder/pan_label/'
% ms_path = '/home/pcl/BaiXu/MyNet/FusionNN/GF-2/Contrast/GPPNN20223172132/TestFolder/ms/'
% ms_label_path = '/home/pcl/BaiXu/MyNet/FusionNN/GF-2/Contrast/GPPNN20223172132/TestFolder/ms_label/'
% 
% lr_fusion_path = '/home/pcl/BaiXu/MyNet/FusionNN/GF-2/Contrast/GPPNN20223172132/FusionFolderLR/'
% hr_fusion_path = '/home/pcl/BaiXu/MyNet/FusionNN/GF-2/Contrast/GPPNN20223172132/FusionFolderHR/'

pan_label_path = '../test_GF2/pan/';
ms_label_path = '../test_GF2/gt/';

lr_fusion_path = '../test_GF2/result/';
hr_fusion_path = '../test_GF2/bic/';

value =  Geteval(ms_label_path,lr_fusion_path,pan_label_path,hr_fusion_path,lr_fusion_path,200,'none');
value_mat(i*2 + 1:i*2 + 2,:) = value;     
disp('     ergas     rmse      rase      qave      sam       ssim      fsim      qnr      D_lambda      D_s');
disp(value)
xlswrite('results_out.xlsx', value_mat);
