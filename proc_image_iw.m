%% Preparation of data file name

prepaname1=truncname(background,'','.tiff','beforelast');
prepaname2=truncname(prepaname1,'C','','afterlast');

if(multimages==0);
prepaname3=truncname(foreground,'','.tiff','beforelast');
else
prepaname3=truncname(foreground{im},'','.tiff','beforelast')  ;  
end;

prepaname4=truncname(prepaname3,'C','','afterlast')

namesavefile=strcat('_',prepaname2,'_',prepaname4);

%% RGB to Mono and image crop

if(multimages==0);
fgimage=imread(strcat(pathname,foreground));
else
fgimage=imread(strcat(pathname,foreground{im}));    
end

fgimage_onecolor=fgimage(:,:,choosecolor); % get the relevant color

fgimage_crop =imcrop(fgimage_onecolor, window); % and crop

figure(1)
subplot(1,2,2)
imshow(fgimage_crop)

%% Absorption calculation

ratio_onecolor = double(fgimage_crop)./double(bgimage_crop);
log_ratio = log(double(fgimage_crop)./double(bgimage_crop));

[m,n]=size(log_ratio);

for ii=1:m
    for jj = 2:n
        if isinf(log_ratio(ii,jj)) == 1
            log_ratio(ii,jj) = log_ratio(ii,(jj-1));
        end
    end
end

mean_ratio_profile=mean(log_ratio,2);
std_ratio_profile=std(mean_ratio_profile,0,2);
intensity=flipud(mean_ratio_profile);

%% Figures

% figure(2)
% 
% subplot(2,3,1)
% imagesc(log((double(bgimage_crop))));
% subplot(2,3,4)
% plot(log((double(bgimage_crop))),'b');
% subplot(2,3,2)
% imagesc(log((double(fgimage_crop))));
% subplot(2,3,5)
% plot(log((double(fgimage_crop))),'b');
% subplot(2,3,3)
% imagesc(log_ratio);
% subplot(2,3,6)
% plot(log_ratio,'b')
% hold on;
% plot(mean_ratio_profile,'r')
% hold off;
% 
% if (choosecolor==1);
%    title('Red component','Fontsize',20,'fontweight','demi')    
% elseif (choosecolor==2);   
%    title('Green component','Fontsize',20,'fontweight','demi')   
% elseif (choosecolor==3);
%    title('Blue component','Fontsize',20,'fontweight','demi')  
% end;


%% Correspondance pixel / m

if askref==0
refpic=input('reference picture (1) or not (0) =>') 
askref=1;
end

if (refpic==1);
figure(3)
imshow(fgimage_onecolor)
input('Position of the free surface, then tank bottom')
[jj,ii]=ginput;

iitop=floor(ii(1)) %position of the free surface
iibot=floor(ii(2)) %position of the tank bottom

imheight=depth*(size(mean_ratio_profile,1))/(iibot-iitop);
height_under_bottom=(size(mean_ratio_profile,1)-iibot)/size(mean_ratio_profile,1)*imheight;

gamma=depth/(iibot-iitop)

zz=imheight*linspace(0,1,size(mean_ratio_profile,1))-height_under_bottom;
end

%% Plot of the raw absorption profile
performspline=0;

while (performspline==0);
clear inten1 zlin1 inten2 zlin2 splintensity
figure(100)
title('Absorption profile','Fontsize',20,'fontweight','demi')
subplot(1,2,1)
%hold on;
plot(intensity,zz,'linewidth',2)
hold off;
axis([min(intensity)-0.5 max(intensity)+0.5 0 depth])
xlabel('ln (Ix/Ix0)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Absorption profile','Fontsize',20,'fontweight','demi')

%% Spline of the intensity profile

if (refpic==1);

input('Linear interpolation of bottom section, start with top point')

slopeornot=input('Constant (1) or varying (2) intensity? ==> ');
[inten1,zlin1]=ginput; 
if (slopeornot==1);
alpha1=0;
else
alpha1=(inten1(1)-inten1(2))/(zlin1(1)-zlin1(2));

iilintop1=find(abs(zz-zlin1(1))-min(abs(zz-zlin1(1)))==0);
iilinbot1=find(abs(zz-zlin1(2))-min(abs(zz-zlin1(2)))==0);
end;
clear slopeornot
%end;

input('Linear interpolation of top section, start with bottom point')

slopeornot=input('Constant (1) or varying (2) intensity? ==> ');
[inten2,zlin2]=ginput;
if (slopeornot==1);
alpha2=0;
else
alpha2=(inten2(1)-inten2(2))/(zlin2(1)-zlin2(2));
end;
iilinbot2=find(abs(zz-zlin2(1))-min(abs(zz-zlin2(1)))==0);

end;

splintensity=intensity;
splintensity(1:iilintop1)=intensity(iilintop1)+alpha1*(zz(1:iilintop1)-zz(iilintop1));
splintensity(iilinbot2:end)=intensity(iilinbot2)+alpha2*(zz(iilinbot2:end)-zz(iilinbot2));



hold on;
plot(splintensity,zz,'r','linewidth',2)
axis([min(splintensity)-0.5 max(splintensity)+0.5 0 depth])
hold off;

if (refpic==1);
performspline=input('Happy with the splintensity profile (0 for no, 1 for yes)? ==>'); % if unhappy, try again.
else
performspline=1;   
end;
end

% save(strcat(pathname,'../results/','intensity',namesavefile),'intensity','splintensity','zz','window','rho1','rho2','dh','dhp');






