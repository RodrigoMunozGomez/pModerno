function imagen = SVD_DECOMP(fileID)
    fileID = fopen(fileID,'rb');
    % color space
    color = fread(fileID,1,"ubit3");
    
    %dimensiones Matrices
    size1U = fread(fileID,2,'int16');
    size2U = fread(fileID,2,'int16');
    size3U = fread(fileID,2,'int16');
    
    size1V = fread(fileID,2,'int16');
    size2V = fread(fileID,2,'int16');
    size3V = fread(fileID,2,'int16');
    %
    % maximos y minimos
    mmU1 = fread(fileID,2,"single");
    mmU2 = fread(fileID,2,"single");
    mmU3 = fread(fileID,2,"single");
    
    mmV1 = fread(fileID,2,"single");
    mmV2 = fread(fileID,2,"single");
    mmV3 = fread(fileID,2,"single");
    %
    
    % Cuantizaciones
    cU1 = fread(fileID,1,'ubit5');
    cU2 = fread(fileID,1,'ubit5');
    cU3 = fread(fileID,1,'ubit5');
    
    cV1 = fread(fileID,1,'ubit5');
    cV2 = fread(fileID,1,'ubit5');
    cV3 = fread(fileID,1,'ubit5');
    %
    
    % Cuantizacion de matrices
    
    U1 = fread(fileID,prod(size1U),'ubit'+string(cU1));
    U2 = fread(fileID,prod(size2U),'ubit'+string(cU2));
    U3 = fread(fileID,prod(size3U),'ubit'+string(cU3));
    
    mins1 = min(size1U(2),size1V(2));
    mins2 = min(size2U(2),size2V(2));
    mins3 = min(size3U(2),size3V(2));

    S1 = fread(fileID,(min(size1U(2),size1V(2))),'single');
    S2 = fread(fileID,(min(size2U(2),size2V(2))),'single');
    S3 = fread(fileID,(min(size3U(2),size3V(2))),'single');
    
    V1 = fread(fileID,prod(size1V),'ubit'+string(cV1));
    V2 = fread(fileID,prod(size2V),'ubit'+string(cV2));
    V3 = fread(fileID,prod(size3V),'ubit'+string(cV3));
    
    fclose(fileID);
    
    U1 = reshape(U1,size1U');
    U2 = reshape(U2,size2U');
    U3 = reshape(U3,size3U');
    
    S1_z = zeros(size1U(2),size1V(2));
    S2_z = zeros(size2U(2),size2V(2));
    S3_z = zeros(size3U(2),size3V(2));
    
    for I = 1:mins1
        S1_z(I,I) = S1(I);
    end
    for I = 1:mins2
        S2_z(I,I) = S2(I);
    end
    for I = 1:mins3
        S3_z(I,I) = S3(I);
    end

    V1 = reshape(V1,size1V');
    V2 = reshape(V2,size2V');
    V3 = reshape(V3,size3V');
    
    % Reconstruir
    
    U1 = decuantizar(U1,mmU1(1),mmU1(2),cU1);
    U2 = decuantizar(U2,mmU2(1),mmU2(2),cU2);
    U3 = decuantizar(U3,mmU3(1),mmU3(2),cU3);
    
    V1 = decuantizar(V1,mmV1(1),mmV1(2),cV1);
    V2 = decuantizar(V2,mmV2(1),mmV2(2),cV2);
    V3 = decuantizar(V3,mmV3(1),mmV3(2),cV3);
    
    imagen(:,:,1) = U1*S1_z*(V1');
    imagen(:,:,2) = U2*S2_z*(V2');
    imagen(:,:,3) = U3*S3_z*(V3');

    if color == 0
        imagen = uint8(imagen);
    elseif color == 1
        imagen = ycbcr2rgb(uint8(imagen));
    elseif color == 2
        imagen = uint8(lab2rgb(imagen)*255.0);
    elseif color == 3
        imagen = uint8(ntsc2rgb(imagen)*255.0);
    elseif color == 4
        imagen = yuv2rgb(imagen);
    end
end
%% funciones
function [matriz] = decuantizar(matriz_cuantizada, min, max, bits)
    num_bins = 2^bits;
    paso = (max - min) / (num_bins - 1);
    matriz = (matriz_cuantizada * paso) + min;
end

function rgb = yuv2rgb(imagen)
    Y = imagen(:, :, 1);
    U = imagen(:, :, 2);
    V = imagen(:, :, 3);
    R = Y + 1.13983 * V;    
    G = Y - 0.39465 * U - 0.5806 * V; 
    B = Y + 2.03211 * U;    
    rgb = cat(3, R, G, B) * 255.0; 
    rgb = uint8(rgb);
end