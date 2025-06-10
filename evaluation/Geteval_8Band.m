function value = Geteval_8Band(labelpath,fusionpath,panpath,mspath,nrfusionpath,N,sensor)
%     discard_list = [
%     8 9 10 11 28 38 39 54 60 68 70 83 84 93 100 103 107 108 113 122 ...
%     129 130 136 137 138 159 160 166 168 169 177 187 188 195 196 197 ...
%     198 199 200 201 213 224 225 226 227 228 229 230 231 232 237 238
%     ]
    % discard_list = [];
    startindex = 0;
    Image_Fusion = zeros([N, 128, 128, 8]);
    Image_Lable = zeros([N, 128, 128, 8]);
    %num_except = 15;
    %% Read Image
    for i = startindex : 1 : N- 1
        for j = 1 : 1 : 8
            Image_Fusion(i+1, :, :, j) = imread(strcat(fusionpath, num2str(j), '/', num2str(i), '.tif'));
            Image_Lable(i+1, :, :, j) = imread(strcat(labelpath, num2str(j), '/', num2str(i), '.tif'));
        end
    end
    %% ERGAS
    ergas = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N- 1
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            ergas(idx) = ERGAS_8Band(GT,MS);
            idx = idx + 1;
%         end
    end
    ergas = ergas(1,1 : idx-1);
    ergas_mean = mean(ergas);
    ergas_var = var(ergas);
    %% RMSE
    rmse = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N- 1
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            rmse(idx) = RMSE(GT,MS);
            idx = idx + 1;
%         end
    end
    rmse = rmse(1,1:idx-1);
    rmse_mean = mean(rmse);
    rmse_var = var(rmse);
    %% RASE
    rase = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            rase(idx) = RASE_8Band(GT,MS);
            idx = idx + 1;
%         end
    end
    rase = rase(1,1:idx-1);
    rase_mean = mean(rase);
    rase_var = var(rase);
    %% QAVE
    qave = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            qave(idx) = QAVE_8Band(GT,MS);
            idx = idx + 1;
%         end
    end
    qave = qave(1,1:idx-1);
    qave_mean = mean(qave);
    qave_var = var(qave);
    %% SAM
    sam = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            [SAM_index,~] = SAM(double(GT),double(MS));
            sam(idx) = SAM_index;
            idx = idx + 1;
%         end
    end
    sam = sam(1,1:idx-1);
    sam_mean = mean(sam);
    sam_var = var(sam);
    %% SSIM
    ssim = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            ssim(idx) = SSIM_8Band(GT,MS);
            idx = idx + 1;
%         end
    end
    ssim = ssim(1,1:idx-1);
    ssim_mean = mean(ssim);
    ssim_var = var(ssim);
    %% FSIM
    fsim = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1 
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(labelpath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            MS = squeeze(Image_Fusion(i+1,:,:,:));
            GT = squeeze(Image_Lable(i+1,:,:,:));
            fsim(idx) = FSIM_8Band(GT,MS);
            idx = idx + 1;
%         end
    end
    fsim = fsim(1,1:idx-1);
    fsim_mean = mean(fsim);
    fsim_var = var(fsim);
%     %% Non-Refference
%     qnr = zeros([1,N]);
%     dlam = zeros([1,N]);
%     ds = zeros([1,N]);
%     idx = 1;
%      for i = startindex : 1 : N - 1 
%         if (~ismember(i,discard_list))
%             MS = imread(strcat(mspath ,num2str(i),'.tif'));
%             Fusion = imread(strcat(nrfusionpath ,num2str(i),'.tif'));
%             PAN = imread(strcat(panpath ,num2str(i),'.tif'));
%             MSUP = imresize(MS,4,'bicubic');
%             [QNR_index,D_lambda_index,D_s_index] = QNR(Fusion,MSUP,PAN,sensor,4,32);
%             qnr(idx) = QNR_index;
%             dlam(idx)  = D_lambda_index;
%             ds(idx)  = D_s_index;
%             idx = idx + 1;
%         end
%      end
%     qnr = qnr(1,1:idx-1);
%     qnr_mean = mean(qnr);
%     qnr_var = var(qnr);
%     dlam = dlam(1,1:idx-1);
%     dlam_mean = mean(dlam);
%     dlam_var = var(dlam);
%     ds = ds(1,1:idx-1);
%     ds_mean = mean(ds);
%     ds_var = var(ds);
    %% Generate Matrix
    value = [
      ergas_mean,rmse_mean,rase_mean,qave_mean,sam_mean,ssim_mean,fsim_mean%,qnr_mean,dlam_mean,ds_mean;
      ergas_var,rmse_var,rase_var,qave_var,sam_var,ssim_var,fsim_var%,qnr_var,dlam_var,ds_var;
    ];    
end