%% Experimental parameters

% Get parameters from xls table

numexp=input('Experiment number ==>');

data=xlsread('/home/dossmann3/turbulent_mixing/4_docs/tabl_exps_iw_2019.xls');


     rhobot=data(numexp,11); %bottom density at exp start
     rhotop=data(numexp,12);  %top density at exp start
     depth=data(numexp,8);    %Fluid depth


%% Background image (Ixo)

[background,pathname] = uigetfile('*.tiff','Choose background image');
window=[];

choosecolor=input('What color: red (1), green (2), blue (3) =>') % choose blue for red dye

bgimage=imread(strcat(pathname,background));

bgimage_onecolor=bgimage(:,:,choosecolor);

if (size(window)==0)
   [bgimage_crop, window] = imcrop(bgimage_onecolor); % Image crop : choose a vertical slice that extends from bottom to top of image
else
   bgimage_crop =imcrop(bgimage_onecolor, window); %window[left top width height]
end

figure(1)
subplot(1,2,1)
imshow(bgimage_crop)

bgimage_crop(:,:)=1; % no background image : in the case of an initially varing stratification

%% Foreground image (Ix)

if choosefore==1;

[foreground,pathname] = uigetfile('*.tiff','Choose foreground image','MultiSelect','on');

if (multimages==0);
    endim=1;
else
    endim=size(foreground,2);
end
end
%% Creating a saving directory

mkdir(strcat(pathname,'/results'));