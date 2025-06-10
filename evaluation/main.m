clear all
clc
close all

N = 1;
value_mat = zeros([N * 2,10]);

pan_label_path = '../test_QB2/pan/';
ms_label_path = '../test_QB2/gt/';

lr_fusion_path = '../test_QB2/result/';
hr_fusion_path = '../test_QB2/bic/';

value =  Geteval(ms_label_path,lr_fusion_path,pan_label_path,hr_fusion_path,lr_fusion_path,200,'none');
value_mat(i*2 + 1:i*2 + 2,:) = value;     
disp('     ergas     rmse      rase      qave      sam       ssim      fsim      qnr      D_lambda      D_s');
disp(value)
xlswrite('results_out.xlsx', value_mat);
