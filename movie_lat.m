%% 


% [infofile,pathname] = uigetfile('*.mat','Choose info file for beta and gamma values');
% load(strcat(pathname,infofile),'beta','gamma');

iback=8114;
ideb=8230;
ifin=8240;

%% Start movie 
film=VideoWriter('it18.avi');
      film.FrameRate=5;
      open(film);
      
      ibackstr=num2str(iback);
      
for it=ideb:1:ifin;
itstr=num2str(it)
    load(strcat('results/densityfields/','density_',ibackstr,'_',itstr,'.mat'))
    
     figure(50)

     Bimg=imagesc(xxc,zzc,-mlratio*beta);
     daspect([1 1 1])
     caxis([-1.5 1.5])
     cmocean('balance')
     colorbar 
     title(strcat('it18, \Delta \rho (kg/m3), t=',num2str((it-ideb)/2.5),' s'));
     xlabel ('x (m)','Fontsize',30);
     ylabel ('z (m)','Fontsize',30);
     set(gca,'Fontsize',30)
     hold off;
     
      MIT=getframe(gcf);
  writeVideo(film,MIT);
  
end;

close(film);
     