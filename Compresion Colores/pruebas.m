addpath("codigo\");
addpath("imagenes\")

k = 10;

imagen = imread("imagenes\journey.jpg");
sz = size(imagen);
imagen_v = reshape(imagen,[],3);
CU = unique(imagen_v,"rows");


tic
COLORCOMP(imagen,k);
[IMG_RECON] = COLORDECOMP;
time = toc;
subplot(1,2,1)
imshow(imagen)

title(sprintf("Original ;CU : %d ; Dim: %d x %d x 3",length(CU),sz(1),sz(2)))

subplot(1,2,2)
imshow(IMG_RECON)
title(sprintf("K = %d ; Tiempo Empleado = %f [Seg];",k,time))