function SVD_COMP(imagen,p1,p2,p3,color_space,name_comp)

    if nargin < 3
        p2 = p1;
        p3 = p1;
    end
    
    if nargin < 5
        color_space = "RBG";
    end
    color_space = lower(color_space);
    if nargin < 6
        name_comp = "imagen_comp_" + color_space + sprintf("_%d_%d_%d.ror",p1,p2,p3);    
    end
    
    if strcmp(color_space,"rgb")
        color = 0;
    elseif strcmp(color_space,"ycbcr")
        imagen  = rgb2ycbcr(imagen);
        color = 1;
    elseif strcmp(color_space,"lab")
        imagen = rgb2lab(imagen);
        color = 2;
    elseif strcmp(color_space,"ntsc")
        imagen = rgb2ntsc(imagen);
        color = 3;
    elseif strcmp(color_space,"yuv")
        imagen = rgb2yuv(imagen);
        color = 4;
    else
        error(color_space + "No es Valido");
    end

    imagen = double(imagen);
    [U1, S1, V1] = svd_mat(imagen(:,:,1),p1);
    [U2, S2, V2] = svd_mat(imagen(:,:,2),p2);
    [U3, S3, V3] = svd_mat(imagen(:,:,3),p3);
    
    %dimensiones Matrices
    size1U = size(U1);
    size2U = size(U2);
    size3U = size(U3);
    size1V = size(V1);
    size2V = size(V2);
    size3V = size(V3);
    %
    
    % maximos y minimos
    mmU1 = min_max(U1);
    mmU2 = min_max(U2);
    mmU3 = min_max(U3);
    
    mmV1 = min_max(V1);
    mmV2 = min_max(V2);
    mmV3 = min_max(V3);
    %
    
    % Cuantizaciones
    cU1 = ceil(log2(length(unique(U1))));
    cU2 = ceil(log2(length(unique(U2))));
    cU3 = ceil(log2(length(unique(U3))));
    
    cV1 = ceil(log2(length(unique(V1))));
    cV2 = ceil(log2(length(unique(V2))));
    cV3 = ceil(log2(length(unique(V3))));
    %
    
    % Cuantizacion de matrices
    
    U1 = cuantizar(U1,mmU1(1),mmU1(2),cU1);
    U2 = cuantizar(U2,mmU2(1),mmU2(2),cU2);
    U3 = cuantizar(U3,mmU3(1),mmU3(2),cU3);
    
    S1 = diag(S1);
    S2 = diag(S2);
    S3 = diag(S3);
    
    V1 = cuantizar(V1,mmV1(1),mmV1(2),cV1);
    V2 = cuantizar(V2,mmV2(1),mmV2(2),cV2);
    V3 = cuantizar(V3,mmV3(1),mmV3(2),cV3);
    %% Guardar
    fileID = fopen(name_comp,'wb');
    % color space
    fwrite(fileID,color,"ubit3");
    % Dimensiones
    fwrite(fileID,size1U,'int16');
    fwrite(fileID,size2U,'int16');
    fwrite(fileID,size3U,'int16');
    
    fwrite(fileID,size1V,'int16');
    fwrite(fileID,size2V,'int16');
    fwrite(fileID,size3V,'int16');
    % Minimos y Maximos 
    fwrite(fileID,mmU1,'single');
    fwrite(fileID,mmU2,'single');
    fwrite(fileID,mmU3,'single');
    
    fwrite(fileID,mmV1,'single');
    fwrite(fileID,mmV2,'single');
    fwrite(fileID,mmV3,'single');
    % Niveles de cuantizacion
    fwrite(fileID,cU1,'ubit5');
    fwrite(fileID,cU2,'ubit5');
    fwrite(fileID,cU3,'ubit5');
    
    fwrite(fileID,cV1,'ubit5');
    fwrite(fileID,cV2,'ubit5');
    fwrite(fileID,cV3,'ubit5');
    % Matrices Cuantizadas
    fwrite(fileID,U1,'ubit'+string(cU1));
    fwrite(fileID,U2,'ubit'+string(cU2));
    fwrite(fileID,U3,'ubit'+string(cU3));
    
    fwrite(fileID,S1,'single');
    fwrite(fileID,S2,'single');
    fwrite(fileID,S3,'single');
    
    fwrite(fileID,V1,'ubit'+string(cV1));
    fwrite(fileID,V2,'ubit'+string(cV2));
    fwrite(fileID,V3,'ubit'+string(cV3));
    fclose(fileID);
end
%% Funciones
function [ambas] = min_max(matriz)
    maximo = max(matriz(:));
    minimo = min(matriz(:));
    ambas = [minimo,maximo];
end
function [matriz_cuantizada] = cuantizar(matriz, min, max, bits)
    num_bins = 2^bits;
    paso = (max - min) / (num_bins - 1);
    matriz_cuantizada = round((matriz - min) / paso);
    matriz_cuantizada(matriz_cuantizada < 0) = 0;
    matriz_cuantizada(matriz_cuantizada > (num_bins - 1)) = num_bins - 1;
end
function[U, Z, V] = svd_mat(imagen,porcentaje)
    [U, Z, V] = pagesvd(double(imagen));
    [m,n,~] = size(imagen);
    s = min(m,n);
    s = round(porcentaje*s/100);
    U = U(:,1:s,:);
    Z = Z(1:s,1:s,:);
    V = V(:,1:s,:);
end

function yuv = rgb2yuv(imagen)
    imagen = double(imagen) / 255.0;
    R = imagen(:, :, 1);
    G = imagen(:, :, 2);
    B = imagen(:, :, 3);
    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    U = -0.14713 * R - 0.288862 * G + 0.436 * B; 
    V = 0.615 * R - 0.51498 * G - 0.10001 * B;  
    yuv = cat(3, Y, U, V);
end