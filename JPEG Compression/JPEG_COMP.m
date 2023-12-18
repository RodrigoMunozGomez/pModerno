% Main
function JPEG_COMP(image, fileName, quality_factor)
    og_size = size(image,1:2);
    
    image = rgb2ycbcr(image);
    image = double(pad(image));

    Y  = image(:,:,1);
    Cb = image(:,:,2);
    Cr = image(:,:,3);

    if quality_factor <= 50
        Cb = imresize(Cb,0.5,"bilinear"); % Subsampling 1/2
        Cr = imresize(Cr,0.5,"bilinear"); % Subsampling 1/2    
    else
        Cb = imresize(Cb,0.5); % Subsampling 1/2
        Cr = imresize(Cr,0.5); % Subsampling 1/2           
    end


    qmatrix_luminance   = qml(quality_factor);
    qmatrix_crominance  = qmc(quality_factor);
   
    Y_q     = jpeg_cuant(Y,qmatrix_luminance);      
    Cb_q    = jpeg_cuant(Cb,qmatrix_crominance);
    Cr_q    = jpeg_cuant(Cr,qmatrix_crominance);

    % % % LUM
    AC_LUM = Y_q;
    DC_LUM = AC_LUM(1:64:end);
    % vector diferencias
    DC_LUM = diff(DC_LUM);
    % primer componente
    DC_LUM = [AC_LUM(1), DC_LUM];
    % quitar dc a ac en lum
    AC_LUM(1:64:end) = 0;
    % % RLE DC Y AC
    AC_LUM = runLengthEncoding(AC_LUM);
    % DC_LUM = runLengthEncoding(DC_LUM); al parecer no conviene 
    
    % % Huffman para AC y DC LUM

    [AC_LUM_CODED, AC_LUM_DICT] = huff(AC_LUM);  % huffman ac luminancia
    profundidad_ac_lum_dict = depth(cell2mat(AC_LUM_DICT(:,1))');
    [DC_LUM_CODED, DC_LUM_DICT] = huff(DC_LUM);  % huffman dc lumniancia testing 
    profundidad_dc_lum_dict = depth(cell2mat(DC_LUM_DICT(:,1))');

    % % % Crom

    AC_CROM = horzcat(Cb_q,Cr_q);
    DC_CROM = AC_CROM(1:64:end);
    % vector diferencias
    DC_CROM = diff(DC_CROM);
    % primer componente
    DC_CROM = [AC_CROM(1),DC_CROM];
    % quitar dc a ac en lum
    AC_CROM(1:64:end) = 0;
    % % RLE DC Y AC
    AC_CROM = runLengthEncoding(AC_CROM);
    %DC_CROM = runLengthEncoding(DC_CROM);
    % % huffman para AC y DC CROM
    [AC_CROM_CODED, AC_CROM_DICT] = huff(AC_CROM);  % huffman ac luminancia
    profundidad_ac_crom_dict = depth(cell2mat(AC_CROM_DICT(:,1))');
    [DC_CROM_CODED, DC_CROM_DICT] = huff(DC_CROM);  % huffman dc lumniancia
    profundidad_dc_crom_dict = depth(cell2mat(DC_CROM_DICT(:,1))');


    jpeg = fopen(fileName + ".rodripeg","wb");
    
    % constantes
    fwrite(jpeg,quality_factor,"uint16");
    fwrite(jpeg,og_size,"uint16");
    sz = size(Y);
    fwrite(jpeg,sz,"uint16");

    % % % LUMINANCE
    % DC
    fwrite(jpeg,profundidad_dc_lum_dict,"ubit5");   % profundidad
    fwrite(jpeg,length(DC_LUM_CODED),"uint32");     % Largo vector codificado     
    fwrite(jpeg,DC_LUM_CODED,"ubit1");
    fwrite(jpeg,size(DC_LUM_DICT,1),"uint16");
    for i = 1:size(DC_LUM_DICT,1)
        fwrite(jpeg,DC_LUM_DICT{i,1},"bit"+string(profundidad_dc_lum_dict));
        fwrite(jpeg,length(DC_LUM_DICT{i,2}), "uint8");   
        fwrite(jpeg,DC_LUM_DICT{i,2},"ubit1");
    end
    % AC
    fwrite(jpeg,profundidad_ac_lum_dict,"ubit5");   % profundidad
    fwrite(jpeg,length(AC_LUM_CODED),"uint32");     % Largo vector codificado     
    fwrite(jpeg,AC_LUM_CODED,"ubit1");
    fwrite(jpeg,size(AC_LUM_DICT,1),"uint16");
    for i = 1:size(AC_LUM_DICT,1)
        fwrite(jpeg,AC_LUM_DICT{i,1},"bit"+string(profundidad_ac_lum_dict));
        fwrite(jpeg,length(AC_LUM_DICT{i,2}), "uint8");   
        fwrite(jpeg,AC_LUM_DICT{i,2},"ubit1");
    end
    % % %CROMINANCE
    % DC
    fwrite(jpeg,profundidad_dc_crom_dict,"ubit5");   % profundidad
    fwrite(jpeg,length(DC_CROM_CODED),"uint32");     % Largo vector codificado     
    fwrite(jpeg,DC_CROM_CODED,"ubit1");
    fwrite(jpeg,size(DC_CROM_DICT,1),"uint16");
    for i = 1:size(DC_CROM_DICT,1)
        fwrite(jpeg,DC_CROM_DICT{i,1},"bit"+string(profundidad_dc_crom_dict));
        fwrite(jpeg,length(DC_CROM_DICT{i,2}), "uint8");   
        fwrite(jpeg,DC_CROM_DICT{i,2},"ubit1");
    end
    % AC
    fwrite(jpeg,profundidad_ac_crom_dict,"ubit5");   % profundidad
    fwrite(jpeg,length(AC_CROM_CODED),"uint32");     % Largo vector codificado     
    fwrite(jpeg,AC_CROM_CODED,"ubit1");
    fwrite(jpeg,size(AC_CROM_DICT,1),"uint16");
    for i = 1:size(AC_CROM_DICT,1)
        fwrite(jpeg,AC_CROM_DICT{i,1},"bit"+string(profundidad_ac_crom_dict));
        fwrite(jpeg,length(AC_CROM_DICT{i,2}), "uint8");   
        fwrite(jpeg,AC_CROM_DICT{i,2},"ubit1");
    end
    fclose(jpeg);
end
% 
function image_dct_q = jpeg_cuant(image, qmatrix)
    image = double(image);
    dct = @(bloque) dct2(bloque.data);
    image_dct = blockproc(image,[8 8],dct);
    quant = @(bloque) round(bloque.data ./qmatrix);
    image_dct_q = blockproc(image_dct, [8 8], quant);
    image_dct_q = subdividir(image_dct_q);
end
function submatrices = subdividir(matriz)
    [filas, columnas] = size(matriz);
    
    subfilas = filas / 8;
    subcolumnas = columnas / 8;

    numSubmatrices = subfilas * subcolumnas;
    submatrices = zeros(8, 8, numSubmatrices);

    k = 1;
    for i = 1:8:filas
        for j = 1:8:columnas
            submatriz = matriz(i:i+8-1, j:j+8-1);
            submatrices(:, :, k) = submatriz;
            k = k + 1;
        end
    end
    submatrices = zigzag(submatrices);
end
function matrix_zigzag_sort = zigzag(matrix)
    zigzag_indices = [  0,   1,  8, 16,  9,  2,  3, 10;
                        17, 24, 32, 25, 18, 11,  4,  5;
                        12, 19, 26, 33, 40, 48, 41, 34;
                        27, 20, 13,  6,  7, 14, 21, 28;
                        35, 42, 49, 56, 57, 50, 43, 36;
                        29, 22, 15, 23, 30, 37, 44, 51;
                        58, 59, 52, 45, 38, 31, 39, 46;
                        53, 60, 61, 54, 47, 55, 62, 63] + 1;
    zigzag_indices = reshape(zigzag_indices.',1,[]);
    [R,C,Channels] = size(matrix);
    matrix_vec = zeros(1,R*C);
    for Channel = 1:Channels
        matrix_vec(Channel,:) = reshape(matrix(:,:,Channel).',1,[]);
        matrix_vec(Channel,:) = matrix_vec(Channel,zigzag_indices);
    end
    matrix_zigzag_sort = reshape(matrix_vec.',1,[]);
end
%
function qmatrix_lum = qml(quality_factor)
    if 1 <= quality_factor && 50>= quality_factor
        alfa = 50/quality_factor;
    elseif 50< quality_factor && 100>= quality_factor
        alfa = 2-(quality_factor/50);
    end
    qmatrix_lum = [
        16  11  10  16  24  40  51  61;
        12  12  14  19  26  58  60  55;
        14  13  16  24  40  57  69  56;
        14  17  22  29  51  87  80  62;
        18  22  37  56  68 109 103  77;
        24  35  55  64  81 104 113  92;
        49  64  78  87 103 121 120 101;
        72  92  95  98 112 100 103  99
    ];
    qmatrix_lum = round(qmatrix_lum * alfa);
    qmatrix_lum(qmatrix_lum == 0) = 1;
end
function qmatrix_crom = qmc(quality_factor)
    if 1 <= quality_factor && 50>= quality_factor
        alfa = 50/quality_factor;
    elseif 50< quality_factor && 100>= quality_factor
        alfa = 2-(quality_factor/50);
    end
    qmatrix_crom = [
        17  18  24  47  99  99  99  99;
        18  21  26  66  99  99  99  99;
        24  26  56  99  99  99  99  99;
        47  66  99  99  99  99  99  99;
        99  99  99  99  99  99  99  99;
        99  99  99  99  99  99  99  99;
        99  99  99  99  99  99  99  99;
        99  99  99  99  99  99  99  99;
    ];
    qmatrix_crom = round(qmatrix_crom * alfa);
    qmatrix_crom(qmatrix_crom == 0) = 1;
end
function rle = runLengthEncoding(datos)
    n = length(datos);
    rle = [];
    count = 1;
    for i = 2:n
        if datos(i) == datos(i-1)
            count = count + 1;
        else
            rle = [rle, datos(i-1), count];
            count = 1;
        end
    end
rle = [rle, datos(n), count];
end
function [vector_codificado, huffman_dict] = huff(piramide_cuant)
    frecuencia = histc(piramide_cuant,unique(piramide_cuant));
    probabilidad = frecuencia/length(piramide_cuant);
    huffman_dict = huffmandict(unique(piramide_cuant), probabilidad);
    vector_codificado = huffmanenco(piramide_cuant, huffman_dict);
end
function paddedImage = pad(image)
    [rows, cols, channels] = size(image);
    padRows = mod(16 - mod(rows, 16), 16);
    padCols = mod(16 - mod(cols, 16), 16);
    paddedImage = zeros(rows + padRows, cols + padCols, channels);
    paddedImage(1:rows, 1:cols, :) = image;
    for i = 1:padRows
        paddedImage(rows + i, 1:cols, :) = image(rows, :, :);
    end
    for j = 1:padCols
        paddedImage(:, cols + j, :) = paddedImage(:,cols,:);
    end
end
function profundidadBits = depth(vec)
    mi = abs(min(vec));
    ma = abs(max(vec)) + 1;
    g = max(mi,ma);
    numero = abs(g);
    if numero == 0
        profundidadBits = 1; 
    else
        profundidadBits = ceil(log2(numero)) + 1;
    end
end