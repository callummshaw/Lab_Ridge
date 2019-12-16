%% Calculation of long term diagnostics : density profile, KT and BPE

clear all


fracdepth=0.5;  
Ltank=4.75; % m; ridge transit distance

%%

numexp=input('Experiment number ==>');

data=xlsread('/home/dossmann3/turbulent_mixing/4_docs/tabl_exps_iw_2019.xls');



    rhobot=data(numexp,11);
    rhotop=data(numexp,12);
     depth=data(numexp,8);
     Uc=data(numexp,9)/(7*60)*0.2513;
     timetrans=Ltank/Uc;

[densityf,pathname] = uigetfile('*.mat','Choose density files','MultiSelect','on');
endim=size(densityf,2)


load(strcat(pathname,densityf{1}))
rhol0=rhol;

ib=find(zz>0);
ib=ib(1);

it=find(zz>depth);
it=it(1)-1;

dz=zz(2)-zz(1);

tabdens=zeros(size(rhol,1),endim);
tabrho=zeros(size(rhol,1),endim);
mass=zeros(1,endim);
drho=zeros(1,endim);
runx=linspace(0,endim-1,endim);

%% Density 


for im=1:endim;
load(strcat(pathname,densityf{im}))
tabdens(:,im)=(rhol(:)-rhol0(:))/(rhobot-rhotop)*100; % Fraction of the max density difference
tabrho(:,im)=rhol(:); % Density profile
mass(1,im)=sum(dz*(rhol(ib:it)-rhol0(ib:it)),1)/sum(dz*rhol0(ib:it),1);
end;

%% Mass correction

for kk=1:size(tabrho,2);

    drho(kk)=(dz/depth)*sum(tabrho(ib:it,kk)-tabrho(ib:it,1)); % Mass correction
    %drho(kk)=tabrho(round(ib+(it-ib)/2),kk)-tabrho(ib+round((it-ib)/2),1)

end;
    tabrhoc=tabrho-drho;
    tabdensc=(tabrhoc-tabrhoc(:,1))/(rhobot-rhotop)*100;

%% Plot Density evolution

figure(106)
subplot(1,2,1)

for kk=1:size(tabrho,2);


hold on;
plot(tabrho(:,kk),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([rhotop-5 rhobot+5 0 depth])

end;

subplot(1,2,2)
imagesc(runx,zz,tabdens)
axis xy
axis ([0 runx(end) 0 depth])
caxis([-10 10])
%xlabel('#run','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Density profile \rho-\rho_0 (%)','Fontsize',20,'fontweight','demi')
cmocean('balance')
colorbar
set(gca,'Fontsize',20,'fontweight','demi')


%% Flux calculation

tabfl=zeros(size(tabdens));
tabk=zeros(size(tabdens));
tabflc=zeros(size(tabdens));
tabkc=zeros(size(tabdens));


ibf=round(ib+fracdepth*(it-ib));

for kk=2:size(tabrho,2);

for ii=ibf:it;
        tabfl(ii,kk)=-1/timetrans*dz*sum(tabrho(ibf:ii,kk)-tabrho(ibf:ii,kk-1));
        tabflc(ii,kk)=-1/timetrans*dz*sum(tabrhoc(ibf:ii,kk)-tabrhoc(ibf:ii,kk-1));
        
        tabk(ii,kk)=-tabfl(ii,kk)*(depth)/(rhotop-rhobot);
        tabkc(ii,kk)=-tabflc(ii,kk)*(depth)/(rhotop-rhobot);
        
end; 
end;

%plot

for kk=1:size(tabrho,2);

figure(101)
subplot(1,2,1)

hold on;
plot(tabrho(:,kk)-tabrho(:,1),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([-5 5 0 depth])

subplot(1,2,2)
hold on;
plot(tabk(:,kk)*10^6,zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('K_t (mm2/ms)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Turbulent diffusivity','Fontsize',20,'fontweight','demi')
axis([min(tabk(:)*10^6) -min(tabk(:)*10^6) 0 depth])

figure(102)
subplot(1,2,1)

hold on;
plot(tabrhoc(:,kk)-tabrhoc(:,1),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([-5 5 0 depth])

subplot(1,2,2)
hold on;
plot(tabkc(:,kk)*10^6,zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('K_t (mm2/ms)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Turbulent diffusivity','Fontsize',20,'fontweight','demi')
axis([min(tabkc(:)*10^6) -min(tabkc(:)*10^6) 0 depth])

end;

%% BPE calculation

tabbpe=zeros(size(tabdens));
tabbpec=zeros(size(tabdens));

g=9.81;
for kk=2:size(tabrho,2);

for ii=ib:it;
        tabbpe(ii,kk)=g*dz*sum((tabrho(ib:ii,kk)-tabrho(ib:ii,1)).*zz(ib:ii).');
        tabbpec(ii,kk)=g*dz*sum((tabrhoc(ib:ii,kk)-tabrhoc(ib:ii,1)).*zz(ib:ii).');

end; 
end;

for kk=1:size(tabrho,2);
    
  figure(103)
subplot(1,2,1)

hold on;
plot(tabrho(:,kk)-tabrho(:,1),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([-5 5 0 depth])

subplot(1,2,2)
hold on;
plot(tabbpe(:,kk),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('BPE (J/s)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('BPE','Fontsize',20,'fontweight','demi')
axis([-max(tabbpe(:)) max(tabbpe(:)) 0 depth])  

figure(104)
subplot(1,2,1)

hold on;
plot(tabrhoc(:,kk)-tabrhoc(:,1),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([-5 5 0 depth])

subplot(1,2,2)
hold on;
plot(tabbpec(:,kk),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('BPE (J/s)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('BPE','Fontsize',20,'fontweight','demi')
axis([min(tabbpec(:)) -min(tabbpec(:)) 0 depth])



end;

%%

save(strcat(pathname,'evoltab'),'zz','tabrho','tabdens','tabrhoc','tabdensc','tabbpec','tabbpe','tabk','tabkc','fracdepth','timetrans','mass')




















