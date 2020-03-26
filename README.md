# Lab Ridge
![Density Anomaly](Pictures/thecapture.PNG)

This a repository of code for analysing lab work with a moving ridge conducted in 2019-2020 at ANU. The code contains some Python scripts and some outdated Jupyter notebooks that were abended for better parallelisation that is achieved with a standard python script (multiprocessing does not seem compatible with Jupyter). 

The code adapts the light attenuation technique used by Yvan Dossman to analyse density variations caused by a moving ridge (with both a tidal and mean flow component), in the large Perspex tank in the Geophysical Fluid Dynamics Laboratory at ANU.

## Installation

Currently the instialltion is very simplistic, download the repository and run the density_and_velocity.py program from the code folder. This will load in the analysis_functions.py as a module and use the functions contained within it. 

## Usage

Running the analysis has been designed to be as straightforward as possible. Just open the analysis_functions.py program. At the top of the program you will see:

```python
excel_path='E:/records.xlsx'
run = 8 
```

Enter the path to your excel document that contains all the information about the runs (density, water depth, etc.) and the experiment run number. After this then run the light_attenuation_analysis function. This function has a number of arguments:

```python
light_attenuation_analysis(run, excel_path, no_hills=1, sigma=0.005, moving_anom = 'no', moving_abs = 'no', fixed_anom = 'no', fixed_abs = 'no', w_vel = 'no', u_vel = 'no'): 
```
The first two arguments are explained above, no_hills is the amount of hills in the topography- currently working for 1 and should work for two hills although this is not yet fully tested. Sigma is the strength of the filter used to generate the velocity fields, the smaller the number the stronger the filter is, the default value is 0.005. The next six arguments are for making animations for various fields (density anomaly and absolute density in both an Eulerian and Lagrangian reference frame). The last two are for making animations for velocity (calculated using the density fields), note currently u_vel is currently not working. If an animation is desired simply change set the desired field equal to 'yes'.

After running the function it will display a number of prompts that will need to be completed. It will first ask for a background image (used as a reference to generate a background density profile). It will then ask you to crop this background image, first click at the location of the free surface then at the bottom of the water column. The program will then generate a background density profile, it will then ask if you are happy with said profile, if so it will continue and if not it will ask you to redo the cropping. 

The program will then ask for the foreground images to analyse. When choosing the foreground images is important to have the topography in all images. It will then produce the videos in the Lagrangian frame if desired. The program will then locate the topography and keep the topography centred in the dataset and produce the density videos in the Eularian frame. The program will then filter the density data to smooth it out in preparation for converting to velocity. To convert from density to velocity the funciton uses the buoyancy equation:

```math
$\frac{D b}{D t}+N^2(z)w\approx 0,$
```
to determine $w$. This yields:

```math
 $w=-\frac{1}{N^2}\frac{D b}{D t},$
 ```
 then use the continuity equation to determine u. 

## Future Additions
