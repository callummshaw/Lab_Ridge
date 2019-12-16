%% Conversion of absorption to density


if (refpic==1);

input('Position of density sample close to bottom, then close to free surface')

[messyr,zsyr]=ginput; %start with bottom
jjbot=find(abs(zz-zsyr(1))-min(abs(zz-zsyr(1)))==0);
jjtop=find(abs(zz-zsyr(2))-min(abs(zz-zsyr(2)))==0);

%% Correspondence intensity/density: case of a linear eos

beta=(rhobot-rhotop)/(splintensity(jjbot)-splintensity(jjtop)); % calculate conversion parameter beta
splinbotref=splintensity(jjbot);

rho=zeros(size(splintensity));
rho=rhobot+beta*(splintensity-splinbotref);
rhoref=rho(1);

rho0=rho;

ib=find(zz>0);
ib=ib(1);

it=find(zz>depth);
it=it(1)-1;

refpic=0;
else

rho=zeros(size(splintensity));
rho=rhobot+beta*(splintensity-splinbotref);

%% Corrections (variation d'intensite)

if (masseq ==0);

rho=rho-rho(1)+rhoref; % equal densities at bottom

elseif (masseq ==1);

dz=zz(2)-zz(1);
rhodecal=sum(dz*(rho(ib:it)-rho0(ib:it)),1)/depth; % egal masses

rho=rho-rhodecal;

end;

end;

subplot(1,2,2)
hold on;
plot(rho,zz,'linewidth',2)
hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([rhotop-5 rhobot+5  0 depth])

rhol=rho;

clear rho

save(strcat(pathname,'/results/','profiles',namesavefile),'window','beta','jjbot','gamma','zz','intensity','splintensity','rhol','iilintop1','iilinbot1')

%gamma: distance conversion
%beta: absorption to density conversion

%% Mean Brunt-Vaisala frequency

% input('Calculation of the average slope')
% 
% [dens,zdens]=ginput; 
% 
% gradrho=(dens(1)-dens(2))/(zdens(1)-zdens(2));
% 
% hold on;
% 
% plot(gradrho*(zz-zdens(1))+dens(1),'linewidth',2,'r')
% axis([rhotop-5 rhobot+5  0 depth])
% 
% N=sqrt(-gradrho/1000*9.81)







