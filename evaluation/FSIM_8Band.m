function N = FSIM_8Band( I_GT,I_F )
%FEATURESIM_8BAND 此处显示有关此函数的摘要
%   此处显示详细说明
I_F_1=I_F(:,:,1);
I_F_2=I_F(:,:,2);
I_F_3=I_F(:,:,3);
I_F_4=I_F(:,:,4);
I_F_5=I_F(:,:,5);
I_F_6=I_F(:,:,6);
I_F_7=I_F(:,:,7);
I_F_8=I_F(:,:,8);

I_GT_1=I_GT(:,:,1);
I_GT_2=I_GT(:,:,2);
I_GT_3=I_GT(:,:,3);
I_GT_4=I_GT(:,:,4);
I_GT_5=I_GT(:,:,5);
I_GT_6=I_GT(:,:,6);
I_GT_7=I_GT(:,:,7);
I_GT_8=I_GT(:,:,8);

fsim_score_1=FeatureSIM(I_F_1,I_GT_1);
fsim_score_2=FeatureSIM(I_F_2,I_GT_2);
fsim_score_3=FeatureSIM(I_F_3,I_GT_3);
fsim_score_4=FeatureSIM(I_F_4,I_GT_4);
fsim_score_5=FeatureSIM(I_F_5,I_GT_5);
fsim_score_6=FeatureSIM(I_F_6,I_GT_6);
fsim_score_7=FeatureSIM(I_F_7,I_GT_7);
fsim_score_8=FeatureSIM(I_F_8,I_GT_8);

N = (fsim_score_1+fsim_score_2+fsim_score_3+fsim_score_4+fsim_score_5+fsim_score_6+fsim_score_7+fsim_score_8)/8;
end

