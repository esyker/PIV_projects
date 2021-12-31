%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Ficheiro de teste
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

path = 
load calib_asus;
depthfiles=['depth_1.mat' 'depth_2.mat'];%change this for different datasets
rgbfiles =['rgb_image_1.png' 'rgb_image_2.png';%change this for different datasets

for i=1:length(depthfiles)
    image_name(i).depth=[path depthfiles(i).name];
    image_name(i).rgb=[path rgbfiles(i).name];
end

[pcloud transforms] = imagematching(image_name, Depth_cam, RGB_cam, R_d_to_rgb, T_d_to_rgb);

