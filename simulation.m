n=A2;
%% 256个环能器重排
m=zeros(128,512);
for j=1:512
       if mod(j,2)==0&&j~=512
           m(1,j)=n(j/2,j/2+1);
       elseif j==512
           m(1,j)=n(j/2,1);
       else
           m(1,j)=n((1+j)/2,(1+j)/2);
       end
end
for i=2:128
    for j=1:512
     t=(1+j)/2+i-1;
     f=j/2+i;
        if mod(j,2)==1
            if i>(1+j)/2
                m(i,j)=n(256-(i-(1+j)/2)+1,(1+j)/2+i-1);
            else
                if t>256
                m(i,j)=n((1+j)/2-i+1,t-256);
                else
                m(i,j)=n((1+j)/2-i+1,t);
                end
            end
        else 
            if i>j/2
                m(i,j)=n(256-(i-j/2)+1,j/2+i);
            else
                if f>256
                m(i,j)=n(j/2-i+1,f-256);
                else
                m(i,j)=n(j/2-i+1,f);  
                end
            end      
         end
     end
end

r=zeros(64,1);
circusR=0.085;
t=circusR;

for j=1:64
    r(j,:)=t-circusR*cos(j*0.0245);
    t=circusR*cos(j*0.0245);
end

r1=flipud(r);
r=[r;r1];
t=0;

for j=1:128
    r(j,:)=t+r(j,:);
    t=r(j,:);
end

r=r';
m=m';
z=linspace(0,2*circusR,128);

for j=1:512
    m(j,:)=spline(r,m(j,:),z);
end
% m=spline(r,m,z);
Z=m';
for i=1:128
    for j=1:512
        if Z(i,j)<0
            Z(i,j)=0;
            continue
        end
    end
end
Z(1,:)=0;
Z(128,:)=0;

%% 稀疏角度FBP成像
l1=1:4:512;%输入4，则从512角度变为128角度；输入8，则从512角度变为64角度
Z11=Z(:,l1);
sinogram_128=Z11;
 theta=1:360/128:360.4;
 fbp=iradon(Z11,theta,128);
 figure;
 imshow(fbp,[0,0.06]);
%% 稀疏角度SART成像
N = 128;
N2 = N ^ 2;
l2=1:2:256;%输入4，则从512角度变为128角度；输入8，则从512角度变为64角度
P=Z;
P=P(:,l2);
Theta=1:180/128:180.4; 
P_num = 128;  % 探测器通道个数

% P = medfuncParallelBeamForwardProjection(theta, N, P_num);   % 产生投影数据
% P = radon(I, theta);
%%===========获取投影矩阵============%%
delta = 1;  % 网格大小
[W_ind, W_dat] = medfuncSystemMatrix(Theta, N, P_num, delta);
%%============设置参数============%%
F = ones(N2, 1);  % 初始图像向量
lambda =0.01;  % 松弛因子
c = 0; % 迭代计数器
% d=zeros(20,1);
irt_num = 100; % 总迭代次数
while( c < irt_num)
    for j = 1:length(Theta)
        W1_ind = W_ind((j - 1) * P_num + 1:j * P_num, :);
        W1_dat = W_dat((j - 1) * P_num + 1:j * P_num, :);
        W = zeros(P_num, N2);
        for jj = 1: P_num
            % 如果射线不经过任何像素，不作计算
            if ~any(W1_ind(jj, :))
                continue;
            end
            for ii = 1:2*N
                m = W1_ind(jj, ii);
                if m > 0 && m <= N2
                    W(jj, m) = W1_dat(jj, ii);
                end
            end
        end
        sumCol = sum(W)'; % 列和向量
        sumRow = sum(W, 2); % 行和向量
        ind1 = sumRow > 0;
        corr = zeros(P_num, 1);
        err = P(:, j) - W * F;
        corr(ind1) = err(ind1) ./ sumRow(ind1);   % 修正误差
        backproj = W' * corr;   % 修正误差反投影
        ind2 = sumCol > 0;
        delta = zeros(N2, 1);
        delta(ind2) = backproj(ind2) ./ sumCol(ind2);
        F = F + lambda * delta;
        F(F < 0) = 0;
    end
    c = c + 1;
   F = reshape(F, N, N)';
%    d(c,:) = psnr(I,F);
   F =reshape(F', N^2, 1);
  
end
   F = reshape(F, N, N)';
   
% figure(1);
% imshow(I), xlabel('(a)180×180头模型图像');
figure;
imshow(F,[0,0.06]);
 
 