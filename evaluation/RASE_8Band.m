function N = RASE_8Band( MS, F )
%RASE_8BAND 此处显示有关此函数的摘要
%   此处显示详细说明
MS=double(MS);
F=double(F);
[m,n,~]=size(F);
% A1=reshape(M1,[m*n,1]);
% A2=reshape(M2,[m*n,1]);
% A3=reshape(M3,[m*n,1]);
% A4=reshape(M4,[m*n,1]);
% A5=reshape(M5,[m*n,1]);
% A6=reshape(M6,[m*n,1]);
% A7=reshape(M7,[m*n,1]);
% A8=reshape(M8,[m*n,1]);
C1=(sum(sum((MS(:,:,1)-F(:,:,1)).^2))/(m*n));
C2=(sum(sum((MS(:,:,2)-F(:,:,2)).^2))/(m*n));
C3=(sum(sum((MS(:,:,3)-F(:,:,3)).^2))/(m*n));
C4=(sum(sum((MS(:,:,4)-F(:,:,4)).^2))/(m*n));
C5=(sum(sum((MS(:,:,5)-F(:,:,5)).^2))/(m*n));
C6=(sum(sum((MS(:,:,6)-F(:,:,6)).^2))/(m*n));
C7=(sum(sum((MS(:,:,7)-F(:,:,7)).^2))/(m*n));
C8=(sum(sum((MS(:,:,8)-F(:,:,8)).^2))/(m*n));
C = C1+C2+C3+C4+C5+C6+C7+C8;
N=((C/8)^(1/2))*100/(mean(mean(mean(MS))));
end

