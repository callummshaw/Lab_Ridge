
iback=8114;
it=8230;

      
ibackstr=num2str(iback);
itstr=num2str(it);

load(strcat('results/densityfields/','density_',ibackstr,'_',itstr,'.mat'))

figure(66)
     imagesc(xxc,zzc,-mlratio*beta)
     daspect([1 1 1])
     caxis([-2 2])
     cmocean('balance')
     colorbar 
     title('\Delta \rho (kg/m3),it07');
     xlabel ('x (m)','Fontsize',30);
     ylabel ('z (m)','Fontsize',30);
     set(gca,'Fontsize',30)