%% Funcion Compresion y Decompresion SVD
clc; clear;
image_path = "images\perros.tif"; %% Imagen a comprimir
color_space = "lab"; %% rgb % YCbCr % lab %  ntsc % yuv %%
file_name = "imagen_comp"; %% Nombre del archivo compreso
imagen = imread(image_path); %% Lectura de la imagen
p1 =13; % Porcentaje Canal 1
p2 =1; % Porcentaje Canal 2
p3 =1; % Porcentaje Canal 3
SVD_COMP(imagen,p1,p2,p3,color_space,file_name); %% Funcion Compresion
imagen_decomp = SVD_DECOMP(file_name);           %% Funcion Decompresion
% Informacion y Plots de Imagenes
peso_comp = peso(file_name);
peso_im = peso(image_path);
psnr_im = psnr(imagen_decomp,imagen);
ssim_im = ssim(imagen_decomp,imagen);
tamanio_fuente = 14; 

subplot(1, 2, 1);
imshow(imagen);
title(sprintf("Imagen Original | Peso : %.0f [Kb]", peso_im));
xlabel(sprintf("Compresion : %.2f",(peso_comp/peso_im)*100))
set(gca, 'FontSize', tamanio_fuente);

subplot(1, 2, 2);
imshow(imagen_decomp);
title(color_space+sprintf(" | C1:%.2f%% | C2:%.2f%% | C3:%.2f%% | Peso : %.6f [Kb]", p1, p2, p3, peso_comp));
set(gca, 'FontSize', tamanio_fuente); 
xlabel(sprintf("PSNR : %.2f", psnr_im));
ylabel(sprintf("SSIM : %.2f", ssim_im));
set(gca, 'FontSize', tamanio_fuente); 

drawnow;
function peso = peso(nombreArchivo)
    if exist(nombreArchivo, 'file') ~= 2
        error('El archivo no existe o no es un archivo válido.');
    end
    infoArchivo = dir(nombreArchivo);
    Bytes = infoArchivo.bytes;
    peso = Bytes / 1024;
end