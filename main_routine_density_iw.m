

close all 

%% Switches

multimages=1; % process a single (0) or a series of images (1)
masseq=2; % choose the mass correction. No mass correction (2)


choosefore=1;
askref=0;

%%

cd '/home/dossmann3/turbulent_mixing/1_optical_density_measurements/';

run /home/dossmann3/turbulent_mixing/2_matlab/parameters_pictures_choice_iw

for im=1:endim; % Loop on image series
run /home/dossmann3/turbulent_mixing/2_matlab/proc_image_iw
run /home/dossmann3/turbulent_mixing/2_matlab/intensity_to_density_iw
end;

clear foreground