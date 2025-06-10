function N = QAVE_8Band( MS, F )
%QAVE_8BAND 此处显示有关此函数的摘要
%   此处显示详细说明
MS=double(MS);
F=double(F);
[~,~,d]=size(F);
MS=MS(:,:,1:d);

MX= mean(MS,3);
MY= mean(F,3);

M1=MS(:,:,1)-MX;
M2=MS(:,:,2)-MX;
M3=MS(:,:,3)-MX;
M4=MS(:,:,4)-MX;
M5=MS(:,:,5)-MX;
M6=MS(:,:,6)-MX;
M7=MS(:,:,7)-MX;
M8=MS(:,:,8)-MX;

P1=F(:,:,1)-MY;
P2=F(:,:,2)-MY;
P3=F(:,:,3)-MY;
P4=F(:,:,4)-MY;
P5=F(:,:,5)-MY;
P6=F(:,:,6)-MY;
P7=F(:,:,7)-MY;
P8=F(:,:,8)-MY;

QX= (1/d-1)*((M1.^2)+(M2.^2)+(M3.^2)+(M4.^2)+(M5.^2)+(M6.^2)+(M7.^2)+(M8.^2));
QY= (1/d-1)*((P1.^2)+(P2.^2)+(P3.^2)+(P4.^2)+(P5.^2)+(P6.^2)+(P7.^2)+(P8.^2));
QXY= (1/d-1)*((M1.*P1)+(M2.*P2)+(M3.*P3)+(M4.*P4)+(M5.*P5)+(M6.*P6)+(M7.*P7)+(M8.*P8));
Q =(d.*((QXY.*MX).*MY))./(((QX+QY).*((MX.^2)+(MY.^2)))+eps);
[m,n]=size(Q);
N = (1/(m*n))*(sum(sum(Q)));
end

