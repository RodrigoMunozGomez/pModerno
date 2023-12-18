clc; close all; clear;

image_path  = "lena.tif";
image_jpeg_path = "lena.jpeg";
image       = imread(image_path);
image_jpeg  = imread(image_jpeg_path);

i = 100;

%%



tic
JPEG_COMP(image,"image",i);
toc



%%
tic
image_rec = JPEG_DECOMP("image.rodripeg");
toc
subplot(1,3,1)
imshow(image)
title(sprintf("Imagen Original || %.3f [Kb]",tam(image_path))) 
subplot(1,3,2)
imshow(image_rec)
title(sprintf("Imagen Reconstruida RODRIPEG || %.3f [Kb]",tam("image.rodripeg")))
xlabel(sprintf("PSNR : %.2f",psnr(image,image_rec)))
subplot(1,3,3)
imshow(image_jpeg); xlabel(sprintf("PSNR : %.2f", psnr(image_jpeg,image))); title(sprintf("Imagen JPG MATLAB || %.3f [Kb]",tam(image_jpeg_path)))
drawnow










%% funciones
function tamano_kb = tam(nombre_archivo)
    info_archivo = dir(nombre_archivo);
    tamano_kb = info_archivo.bytes / 1024;
end


