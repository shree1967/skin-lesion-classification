
clc;
close all;
clear all;
%------------ Image Reading ------------------------------------------
delete Ndata.mat;
FilePathStorage=pwd;
FilePathname1=strcat(FilePathStorage,'/melanoma/');
FiledirName1=dir([FilePathname1 '*.jpg']);
TotalImages1=size(FiledirName1,1);
Train_dataN=zeros(TotalImages1,286);
Train_LabN=zeros(TotalImages1,1);

for PPP=1:TotalImages1
Filename1=FiledirName1(PPP).name;
FilePathDatabase1=[FilePathname1,Filename1];
[DataArray,map]=imread(FilePathDatabase1);
% Downsample
% DataArray=imresize(DataArray,0.9);
% Otsu's Segmentation
R=DataArray(:,:,1);
G=DataArray(:,:,2);
B=DataArray(:,:,3);
Igray = 0.30*R + 0.59*G + 0.11*B;
% figure,imshow(Igray);
% title('Gray Image');
Img=double(Igray);
timestep=5;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=5;
iter_outer=60;
lambda=5; % coefficient of the weighted length term L(phi)
alfa=1.5;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function
sigma=1.5;     % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma);
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
% figure,imshow(uint8(Img_smooth));
% title('Guassian Filter Output');
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
% figure,imshow(uint8(f));
% title('Gradient Output');
g=1./(1+f);  % edge indicator function.
% initialize LSF as binary step function
c0=2;
initialLSF=c0*ones(size(Img));
% generate the initial region R0 as a rectangle
initialLSF(9:187, 9:175)=-c0;  
phi=initialLSF;
% figure(1);
% mesh(-phi);   % for a better view, the LSF is displayed upside down
% hold on;  contour(phi, [0,0], 'r','LineWidth',2);
% title('Initial level set function');
% view([-80 35]);
% figure;
% imshow(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
% title('Initial zero level contour');
% pause(0.5);
potential=2;  
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
elseif potential == 2
    potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
else
    potentialFunction = 'double-well';  % default choice of potential function
end
% start level set evolution
figure
for n=1:iter_outer
    phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
    if mod(n,2)==0
        imshow(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
        title('DRLSE')
        pause(0.2)
    end
end
bwimg=zeros(size(phi,1),size(phi,2));
for ii=1:size(phi,1)
    for jj=1:size(phi,2)
        if(phi(ii,jj)<=0)
            bwimg(ii,jj)=1;
        else
            bwimg(ii,jj)=0;
        end
    end
end
[L, num1] = bwlabel(bwimg,4);
count_pixels_per_obj = sum(bsxfun(@eq,L(:),1:num1));
[~,ind] = sort(count_pixels_per_obj,'descend');
FSeg = (L==ind(1));
FSeg(FSeg>0)=1;
% figure,imshow(uint8(FSeg*255))
% title('Segmented Output');
% 
Segimg=zeros(size(FSeg,1),size(FSeg,2));
for ii=1:size(FSeg,1)
    for jj=1:size(FSeg,2)
        if(FSeg(ii,jj)<=0)
            Segimg(ii,jj)=0;
        else
            Segimg(ii,jj)=Igray(ii,jj);
        end
    end
end
% figure,imshow(uint8(Segimg))
% title('Segmented Lesion Output');
ROI=regionprops(FSeg,'BoundingBox','Image','Centroid','Solidity','Perimeter','Area','ConvexArea');
NR=vertcat(ROI.BoundingBox);
Area1=zeros(1,size(NR,1));
% FAdata=zeros(1,size(NR,1));
Asym=zeros(1,size(NR,1));
Compactness=zeros(1,size(NR,1));
for ii=1:size(NR,1)
    [H,W]=size(ROI(ii).Image);
    Centroid=ROI(ii).Centroid;
    Adata=1-(2.*sqrt((Centroid(1)-(W/2)).^2 + (Centroid(1)-(H/2)).^2)).^4;
    Area1(1,ii)=Adata;
    Compactness(ii)=(4*ROI(ii).Area*pi)/(ROI(ii).Perimeter).^2;
    Img1=ROI(ii).Image;
    Ix=ROI(ii).Area;Iy=ROI(ii).Area;
    Ix(:,1:end-1)=ROI(ii).Area(:,2:end)-ROI(ii).Area(:,1:end-1);
    Iy(1:end-1,:) = ROI(ii).Area(2:end,:) - ROI(ii).Area(1:end-1,:);
    Asym(1,ii)=(Ix+Iy)./ROI(ii).Area;
    IRdata = imcrop(DataArray,[NR(ii,1) NR(ii,2) NR(ii,3) NR(ii,4)]);
    Fdata=IRdata;
    for i=1:H
    for j=1:W
        if(Img1(i,j)>0)
            Fdata(i,j,1)=IRdata(i,j,1);
            Fdata(i,j,2)=IRdata(i,j,2);
            Fdata(i,j,3)=IRdata(i,j,3);
        else
           Fdata(i,j,1)=0;
           Fdata(i,j,2)=0;
           Fdata(i,j,3)=0; 
        end
    end
    end
end
[~,idx]= sort(Area1,'descend');
ROI_Img=ROI(idx(1)).Image;
Solidity=ROI(idx(1)).Solidity;
convexity=ROI(idx(1)).ConvexArea;
Asymdata=Asym(1,idx(1));
FAdata=ROI(idx(1)).Area;
Compact=Compactness(1,idx(1));
vardata=std(Area1(1,idx(1)));
% figure,imshow(ROI_Img);
% title('Lesion Border Localization');
Skdata=Fdata;
% Feature Extraction
R=Skdata(:,:,1);
G=Skdata(:,:,2);
B=Skdata(:,:,3);
hsvImg=rgb2hsv(Skdata);
h=hsvImg(:,:,1);
s=hsvImg(:,:,2);
v=hsvImg(:,:,3);
Igray=rgb2gray(Skdata);
Mean_Feat=[mean2(R) mean2(G) mean2(B) std2(R) std2(G) std2(B) mean2(h) mean2(s) mean2(v) std2(h) std2(s) std2(v) mean2(Igray) mean2(Igray) mean2(Igray) std2(Igray) std2(Igray) std2(Igray)];
RA=imhist(R);
RB = RA(RA ~= 0);
GA=imhist(G);
GB = GA(GA ~= 0);
BA=imhist(B);
BB = BA(BA ~= 0);
hA=imhist(h);
hB = hA(hA ~= 0);
sA=imhist(s);
sB = sA(sA ~= 0);
vA=imhist(v);
vB = vA(vA ~= 0);
Igray1=imhist(Igray);
grayB = Igray1(Igray1~= 0);
Histdata=[length(RB) length(GB) length(BB) length(hB) length(sB) length(vB) length(grayB)];
LCF=[Mean_Feat Histdata];
RGB = R + G + B;
CTF = mean2(mean2(RGB))/3;
LBF=[Solidity convexity Compact vardata]/1000;
% LBP Feature
Input_Im=Igray;
R=1;
L = 2*R + 1; 
C = round(L/2);
Input_Im = uint8(Input_Im);
row_max = size(Input_Im,1)-L+1;
col_max = size(Input_Im,2)-L+1;
LBP_Im = zeros(row_max+2, col_max+2);
for i = 1:row_max
    for j = 1:col_max
        A = Input_Im(i:i+L-1, j:j+L-1);
        A = A+1-A(C,C);
        A(A>0) = 1;
        LBP_Im(i,j) = A(C,L) + A(L,L)*2 + A(L,C)*4 + A(L,1)*8 + A(C,1)*16 + A(1,1)*32 + A(1,C)*64 + A(1,L)*128;
    end;
end
% figure,imshow(uint8(LBP_Im))
% title('LBP Image')
Lbp_Img=imhist(LBP_Im);
Lbp_Img=Lbp_Img';
disp('Periphery Feature')
disp([LCF LBF Asymdata])
disp('Texture Feature')
disp(Lbp_Img)
Tfeat=[LCF LBF Asymdata Lbp_Img]/255;
disp('Total Features');
disp(Tfeat)
Train_dataN(PPP,:)=Tfeat;
Train_LabN(PPP,1)=2;
end
save Ndata.mat Train_dataN Train_LabN
