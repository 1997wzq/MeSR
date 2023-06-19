% AA=imread('afterSART.png');
% AA=im2gray(AA);
AA=load('sartfangzhen.mat');
AA=AA.F;
% BB=imread('afterSLFBP.png');
% BB=im2gray(BB);
BB=load('fbpfangzhen.mat');
BB=BB.fbp;
%% 一维变换
[cA1,cH1,cV1,cD1]=dwt2(AA,'db1');
[BcA1,BcH1,BcV1,BcD1]=dwt2(BB,'db1');
%% 二维变换系数
[cA,sA]=wavedec2(AA,2,'db1');%使用haar小波
siz=sA(size(sA,1),:);
cA2=appcoef2(cA,sA,'db1',2);
cH2=detcoef2('h',cA,sA,2);
cV2=detcoef2('v',cA,sA,2);
cD2=detcoef2('d',cA,sA,2);
[cB,sB]=wavedec2(BB,2,'db1');%使用haar小波
sizB=sB(size(sB,1),:);
BcA2=appcoef2(cB,sB,'db1',2);
BcH2=detcoef2('h',cB,sB,2);
BcV2=detcoef2('v',cB,sB,2);
BcD2=detcoef2('d',cB,sB,2);

%% 插值到一维变换的尺寸
IA2=upcoef2('a',cA2,'db1',1,[128,128]);
IH2=upcoef2('h',cH2,'db1',1,[128,128]);
IV2=upcoef2('v',cV2,'db1',1,[128,128]);
ID2=upcoef2('d',cD2,'db1',1,[128,128]);

BIA2=upcoef2('a',BcA2,'db1',1,[255,255]);
BIH2=upcoef2('h',BcH2,'db1',1,[255,255]);
BIV2=upcoef2('v',BcV2,'db1',1,[255,255]);
BID2=upcoef2('d',BcD2,'db1',1,[255,255]);
%% 求残差图像并加到各个分量上
residual_imageA=cA1-IA2;%残差图像

IH2=residual_imageA+IH2;
IV2=residual_imageA+IV2;
ID2=residual_imageA+ID2;

residual_imageB=BcA1-BIA2;%残差图像

BIH2=residual_imageB+BIH2;
BIV2=residual_imageB+BIV2;
BID2=residual_imageB+BID2;

%% 得到字典
[cH2H,cH2L]=chuli(cH1,IH2);
[cV2H,cV2L]=chuli(cV1,IV2);
[cD2H,cD2L]=chuli(cD1,ID2);
%% 测试阶段
IH=xishuxishu(BcH1,cH2H,cH2L);
IV=xishuxishu(BcV1,cV2H,cV2L);
ID=xishuxishu(BcD1,cD2H,cD2L);
A=idwt2(BcA1,IH,IV,ID,'db1');

