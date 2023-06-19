function [ResultIM]=xishuxishu(ImTest,Ah,Al)
clear  Eh Permute Ph Ph_hat Ptilde_l Pl Q R SS Yout param
clear output counter patch pos k1 posY0 energy N k2
clear Weight Yh Yl Yl1 Yl2 Yl3 Yl4 Zl posX posY posX0 
n=5; % block size
m=500; % number of atoms in the dictionary
s=2; % scale-down factor
dd=3; % margins in the image to avoid (dd*s to each side)
L=3; % number of atoms to use in the representation
n2=(n-1)/2; 
% Yh=ImTest;

% N=floor(N/s)*s; 
% Yh=im2double(Yh(1:N,1:N)); % so that it dcales down to an integer size
% Yh=Yh*255; 
% IdealIM=Yh; 

% % Creating the low-resolution image
% Zl=conv2(Yh,[0.1 0.3 0.4 0.3 0.1]/1.2,'same');
% Zl=conv2(Zl,[0.1 0.3 0.4 0.3 0.1]'/1.2,'same');
% Zl=Zl(1:s:end,1:s:end); 
% TestIM=Zl; 

% Upscaling Zl to the original resolution
% [posY,posX]=meshgrid(1:s:N,1:s:N); 
% [posY0,posX0]=meshgrid(1:1:N,1:1:N); 
% Yl=interp2(posY,posX,Zl,posY0,posX0,'bicubic');
% TestIMint=Yl; 
Yl=ImTest;
N=size(Yl,1);
N=floor(N/s)*s; 
Yl=im2double(Yl(1:N,1:N));
% Yl=Yl*255;
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
R=Ptilde_l*Ptilde_l'; 
[B,SS]=eig(R); 
Permute=fliplr(eye(size(R,1))); 
SS=Permute*SS*Permute; % so that the eigenvalues are sorted descending
B=B*Permute; 
energy=cumsum(diag(SS))/sum(diag(SS)); 
% figure(1); clf; plot(energy)
pos=find(energy>0.999,1);
pos=10;
B=B(:,1:pos);
Pl=B'*Ptilde_l; 
qq1=Al'*Pl;
qq2=Al'*Al;

% Sparse coding of the low-res patches
Q=omp(qq1,qq2,L); 

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
end