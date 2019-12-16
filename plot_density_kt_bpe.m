%% Plot of long term diagnostics : density profile, KT and BPE

clear all
%close all

masscorr=1;
fracdepth=0.5;
Ltank=4.75;
coarse=1;

%% 

numexp=input('Experiment number ==>');

data=xlsread('/home/dossmann3/turbulent_mixing/4_docs/tabl_exps_iw_2019.xls');



    rhobot=data(numexp,11);
    rhotop=data(numexp,12);
     depth=data(numexp,8);
     Uc=data(numexp,9)/(7*60)*0.2513;
     timetrans=Ltank/Uc;


load('evoltab.mat');


ib=find(zz>0);
ib=ib(1);

it=find(zz>depth);
it=it(1)-1;

dz=zz(2)-zz(1);
endim=size(tabrho,2);



%% No mass correction

if masscorr==0;
 

runx=linspace(0,endim-1,round(endim/coarse));


figure(100)
subplot(1,2,1)


for kk=1:coarse:size(tabrho,2);

hold on;
plot(tabrho(:,kk),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
%hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([rhotop-5 rhobot+5 0 depth])

end;

subplot(1,2,2)
imagesc(runx(1:end),zz,tabdens(:,1:end))
axis xy
axis ([runx(1) runx(end) 0 depth])
caxis([-10 10])
xlabel('#run','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Density profile \rho-\rho_0 (%)','Fontsize',20,'fontweight','demi')
cmocean('balance')
colorbar
set(gca,'Fontsize',20,'fontweight','demi')


%% Flux plot

for kk=1:coarse:size(tabrho,2);
    
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

end;


%% BPE plot


for kk=1:coarse:size(tabrho,2);

figure(102)
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

end;

else

%% Density plot
 

runx=linspace(0,endim-1,round(endim/coarse));


figure(200)
subplot(1,2,1)


for kk=1:coarse:size(tabrhoc,2);

hold on;
plot(tabrhoc(:,kk),zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
%hold off;
xlabel('\rho(kg/m3)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('density profile','Fontsize',20,'fontweight','demi')
axis([rhotop-5 rhobot+5 0 depth])

end;

subplot(1,2,2)
imagesc(runx(1:end),zz,tabdensc(:,1:end))
axis xy
axis ([runx(1) runx(end) 0 depth])
caxis([-10 10])
xlabel('#run','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Density profile \rho-\rho_0 (%)','Fontsize',20,'fontweight','demi')
cmocean('balance')
colorbar
set(gca,'Fontsize',20,'fontweight','demi')


%% Flux plot

for kk=1:coarse:size(tabrho,2);
    
figure(201)
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
plot(tabk(:,kk)*10^6,zz,'linewidth',2,'Color',[kk/endim 0 1-kk/endim])
hold off;
xlabel('K_t (mm2/ms)','Fontsize',20,'fontweight','demi')
ylabel('z (m)','Fontsize',20,'fontweight','demi')
set(gca,'Fontsize',20,'fontweight','demi')
title('Turbulent diffusivity','Fontsize',20,'fontweight','demi')
axis([min(tabk(:)*10^6) -min(tabk(:)*10^6) 0 depth])

end;


%% BPE plot


for kk=1:coarse:size(tabrhoc,2);

figure(202)
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

end;






















