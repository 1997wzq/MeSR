% Figures - 15.27 - 15.31
% =========================================
% This program presents treates the inpainting problem locally as 
% the two previous algorithms, but introduces a leearned dictionary. 
% The results built here are used to fill Table 15.2


% First test - with a text image
%（训练阶段图片，测试阶段图片）
[Train,TestL,TestH,Ideal,Result]=...
                  Chapter_15_ScaleUp_Training_and_Testing1('afterSART.png','afterSLFBP.png');
%CChapter_15_ScaleUp_Training_and_Testing1
%('TextImage3.png',   '
%TextImage4.png')('afterSART.png','afterSLFBP.png')original.png
close all; 


figure(1);
image(Train); axis image; axis off; truesize; colormap(gray(256));

figure(2);
image(TestL); axis image; axis off; truesize; colormap(gray(256));

figure(3);
image(TestH); axis image; axis off; truesize; colormap(gray(256));

figure(4);
image(Ideal); axis image; axis off; truesize; colormap(gray(256));

figure(5);
image(Result); axis image; axis off; truesize; colormap(gray(256));
% score=quality_predict(Result);
figure(6);
image(Result(210:310,210:310));axis image; axis off; truesize;colormap(gray(256));
figure(7);
image(Result(360:460,205:305));axis image; axis off; truesize;colormap(gray(256));

% imwrite(kuai,'H:\xishuyu\code\Matlab-Package-Book\结果图\Minekuai1.png');
% imwrite(kuai2,'H:\xishuyu\code\Matlab-Package-Book\结果图\Minekuai2.png');
% %imwrite(TestH,'H:\xishuyu\code\Matlab-Package-Book\结果图\双三次样条插值s=4.png');
% imwrite(Result,'H:\xishuyu\code\Matlab-Package-Book\结果图\Mine.png');
% O1=uint8(phantom2(510));
O1=imread('H:\xishuyu\code\Matlab-Package-Book\afterSLFBP.png');
O2=imread('H:\xishuyu\code\Matlab-Package-Book\afterSART.png');
O2=uint8(O2);
b=phantom(510);
b=uint8(b * 255);
O1=uint8(O1);
I1=uint8(Ideal);
R1=uint8(Result);
score=avg_gradient(R1);
% b=psnr(I1,R1);
% c=vifvec(O1,R1);%
b1=psnr(O2,b);



% figure(1); clf; 
% Train(:,1)=0; Train(:,end)=0; 
% Train(1,:)=0; Train(end,:)=0;
% image(Train); axis image; axis off; truesize; colormap(gray(256)); 
% % print -depsc2 Chapter_15_ScaleUp_Text_TrainImage.eps
% 
% figure(2); clf; 
% TestL(:,1)=0; TestL(:,end)=0; 
% TestL(1,:)=0; TestL(end,:)=0;
% image(TestL); axis image; axis off; truesize; colormap(gray(256)); 
% % print -depsc2 Chapter_15_ScaleUp_Text_TestLImage.eps
% 
% figure(3); clf; 
% TestH(:,1)=0; TestH(:,end)=0; 
% TestH(1,:)=0; TestH(end,:)=0;
% image(TestH); axis image; axis off; truesize; colormap(gray(256)); 
% % print -depsc2 Chapter_15_ScaleUp_Text_TestHImage.eps
% 
% figure(4); clf; 
% Ideal(:,1)=0; Ideal(:,end)=0; 
% Ideal(1,:)=0; Ideal(end,:)=0;
% image(Ideal); axis image; axis off; truesize; colormap(gray(256)); 
% % print -depsc2 Chapter_15_ScaleUp_Text_TestIdealImage.eps
% 
% figure(5); clf; 
% Result(:,1)=0; Result(:,end)=0; 
% Result(1,:)=0; Result(end,:)=0;
% image(Result); axis image; axis off; truesize; colormap(gray(256)); 
% print -depsc2 Chapter_15_ScaleUp_Text_ResultImage.eps

% % Second test - Child
% 
% Chapter_15_ScaleUp_Training_and_Testing2('BuildingImage1.png'); 
% 
% figure(1); 
% % print -depsc2 Chapter_15_ScaleUp_Building.eps
% 
% figure(2); 
% % print -depsc2 Chapter_15_ScaleUp_Building_Part1.eps
% 
% figure(3); 
% % print -depsc2 Chapter_15_ScaleUp_Building_Part2.eps

