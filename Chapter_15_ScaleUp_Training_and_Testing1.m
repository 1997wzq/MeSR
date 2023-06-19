function [TrainIM,TestIM,TestIMint,IdealIM,ResultIM]=...
            Chapter_15_ScaleUp_Training_and_Testing1(ImTrain,ImTest)

%============================================
%其中字典大小设置为1，块大小设置为9，放大倍数设置为2，训练样本是含有噪声的SART重建图像，测试样本,255*255的低精度FBP重建图像。                              Training the two dictionaries
%============================================

% Setting parameters
n=9; % block size9
m=120; % number of atoms in the dictionary1000
s=2; % scale-down factor3
dd=2; % margins in the image to avoid (dd*s to each side)
L=20; % number of atoms to use in the representation

% Preparing the low-and high resolution images
% Yh=imread(ImTrain);
Yh=ImTrain;
N=size(Yh,1);
N=floor(N/s)*s; 
Yh=im2double(Yh(1:N,1:N)); % so that it dcales down to an integer size
Yh=Yh*255;
TrainIM=Yh; 

% Creating the low-resolution image
Zl=conv2(Yh,[0.1 0.3 0.4 0.3 0.1]/1.2,'same');
Zl=conv2(Zl,[0.1 0.3 0.4 0.3 0.1]'/1.2,'same');
Zl=Zl(1:s:end,1:s:end); 

% Upscaling Zl to the original resolution
[posY,posX]=meshgrid(1:s:N,1:s:N); 
[posY0,posX0]=meshgrid(1:1:N,1:1:N); 
Yl=interp2(posY,posX,Zl,posY0,posX0,'cubic');
Eh=Yh-Yl;
%%
Yl1=conv2(Yl,[0,0,1,0,0,-1],'same'); % the filter is centered and scaled well for s=3
Yl2=conv2(Yl,[0,0,1,0,0,-1]','same');
Yl3=conv2(Yl,[1,0,0,-2,0,0,1]/2,'same');
Yl4=conv2(Yl,[1,0,0,-2,0,0,1]'/2,'same');

% Gathering the patches
Ph=zeros(n^2,(N/s-2*dd)^2); 
Ptilde_l=zeros(4*n^2,(N/s-2*dd)^2); 
n2=(n-1)/2; 
counter=1;
for k1=s*dd+1:s:N-s*dd
    for k2=s*dd+1:s:N-s*dd
        Ph(:,counter)=reshape(Eh(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]);
        Ptilde_l(:,counter)=[reshape(Yl1(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]); ...
                                    reshape(Yl2(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]);...
                                    reshape(Yl3(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]);...
                                    reshape(Yl4(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1])];
        counter=counter+1;
    end;
end;

% Dimentionalily reduction
R=Ptilde_l*Ptilde_l'; 
[B,SS]=eig(R); %B324*324,SS324*324,R324*324
Permute=fliplr(eye(size(R,1))); 
SS=Permute*SS*Permute; % 使得特征值按降序排序
B=B*Permute; 
energy=cumsum(diag(SS))/sum(diag(SS)); 
% figure(1); clf; plot(energy)
pos=find(energy>0.999,1);%30
B=B(:,1:pos);%    324*324---->324*30
disp(['Effective dimension: ',num2str(pos)]); 
disp(['The relative error is: ',...
        num2str(mean(sum((B*B'*Ptilde_l-Ptilde_l).^2))/...
        mean(sum((Ptilde_l).^2)))]); % showing the relative error
Pl=B'*Ptilde_l; %324*54289---->30*54289

% Low-Res. Dictionary Learning 
param.errorFlag=0;
param.K=m; 
param.numIteration=40; 
param.InitializationMethod='DataElements'; 
param.TrueDictionary=0;
param.Method='KSVD';
param.L=L;   
[Al,output]=Chapter_12_TrainDic_Fast(Pl,param);
Q=output.CoefMatrix; 

% High-Resolution Dictionary Leanring
Ah=Ph*Q'*inv(Q*Q');

%============================================
%                  Sanity Check - Interpolating the training image
%============================================

Ph_hat=Ah*Q; 
Yout=Yl*0; 
Weight=Yl*0;
counter=1;
for k1=s*dd+1:s:N-s*dd
    for k2=s*dd+1:s:N-s*dd
        patch=reshape(Ph_hat(:,counter),[n,n]);
        Yout(k1-n2:k1+n2,k2-n2:k2+n2)=...
            Yout(k1-n2:k1+n2,k2-n2:k2+n2)+patch; 
        Weight(k1-n2:k1+n2,k2-n2:k2+n2)=...
            Weight(k1-n2:k1+n2,k2-n2:k2+n2)+1; 
        counter=counter+1;
    end;
end;
Yout=Yout./(Weight+1e-5)+Yl; 
figure(2); clf; imagesc([Yh,Yl,Yout]);
colormap(gray(256)); axis image; axis off; 

ErrOut=mean(mean((Yout(dd*s+1:end-dd*s,dd*s+1:end-dd*s)...
                           -Yh(dd*s+1:end-dd*s,dd*s+1:end-dd*s)).^2));
ErrIn=mean(mean((Yl(dd*s+1:end-dd*s,dd*s+1:end-dd*s)...
                           -Yh(dd*s+1:end-dd*s,dd*s+1:end-dd*s)).^2)); 
disp('Result on the train image: ');
disp([sqrt(ErrIn),sqrt(ErrOut),10*log10(ErrIn/ErrOut)]);

% The only variable to keep:Al, Ah, B, and (dd,L,n,n2,m,s)
clear  Eh Permute Ph Ph_hat Ptilde_l Pl Q R SS Yout param
clear output counter patch pos k1 posY0 energy N k2
clear Weight Yh Yl Yl1 Yl2 Yl3 Yl4 Zl posX posY posX0 

%============================================
%                                Interpolating a test image
%============================================
% Yh=imread(ImTest);        
Yh=ImTest;
N=size(Yh,1);
N=floor(N/s)*s; 
Yh=im2double(Yh(1:N,1:N)); % so that it dcales down to an integer size
Yh=Yh*255; 
IdealIM=Yh; 

% Creating the low-resolution image
Zl=conv2(Yh,[0.1 0.3 0.4 0.3 0.1]/1.2,'same');
Zl=conv2(Zl,[0.1 0.3 0.4 0.3 0.1]'/1.2,'same');
Zl=Zl(1:s:end,1:s:end); 
TestIM=Zl; 

% Upscaling Zl to the original resolution
[posY,posX]=meshgrid(1:s:N,1:s:N); 
[posY0,posX0]=meshgrid(1:1:N,1:1:N); 
Yl=interp2(posY,posX,Zl,posY0,posX0,'bicubic');
TestIMint=Yl; 

% Extracting features
Yl1=conv2(Yl,[0,0,1,0,0,-1],'same'); 
Yl2=conv2(Yl,[0,0,1,0,0,-1]','same');
Yl3=conv2(Yl,[1,0,0,-2,0,0,1]/2,'same');
Yl4=conv2(Yl,[1,0,0,-2,0,0,1]'/2,'same');

% Gathering the patches
Ptilde_l=zeros(4*n^2,(N/s-2*dd)^2); 
counter=1;
for k1=s*dd+1:s:N-s*dd
    for k2=s*dd+1:s:N-s*dd
        Ptilde_l(:,counter)=[reshape(Yl1(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]); ...
                                    reshape(Yl2(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]);...
                                    reshape(Yl3(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1]);...
                                    reshape(Yl4(k1-n2:k1+n2,k2-n2:k2+n2),[n^2,1])];
        counter=counter+1;
    end;
end;

% Dimentionalily reduction
Pl=B'*Ptilde_l; 

% Sparse coding of the low-res patches
Q=omp(Al'*Pl,Al'*Al,L); 

% Recover the image
Ph_hat=Ah*Q; 
Yout=Yl*0; 
Weight=Yl*0;
counter=1;
for k1=s*dd+1:s:N-s*dd
    for k2=s*dd+1:s:N-s*dd
        patch=reshape(Ph_hat(:,counter),[n,n]);
        Yout(k1-n2:k1+n2,k2-n2:k2+n2)=...
            Yout(k1-n2:k1+n2,k2-n2:k2+n2)+patch; 
        Weight(k1-n2:k1+n2,k2-n2:k2+n2)=...
            Weight(k1-n2:k1+n2,k2-n2:k2+n2)+1; 
        counter=counter+1;
    end;
end;
Yout=Yout./(Weight+1e-5)+Yl; 
Yout=min(max(Yout,0),255);
ResultIM=Yout; 
figure(3); clf; imagesc([Yh,Yl,Yout]);
colormap(gray(256)); axis image; axis off; 
truesize; 

ErrOut=mean(mean((Yout(dd*s+1:end-dd*s,dd*s+1:end-dd*s)...
                           -Yh(dd*s+1:end-dd*s,dd*s+1:end-dd*s)).^2));
ErrIn=mean(mean((Yl(dd*s+1:end-dd*s,dd*s+1:end-dd*s)...
                           -Yh(dd*s+1:end-dd*s,dd*s+1:end-dd*s)).^2)); 
disp('Result on the test image: ');
disp([sqrt(ErrIn),sqrt(ErrOut),10*log10(ErrIn/ErrOut)]);

return; 











