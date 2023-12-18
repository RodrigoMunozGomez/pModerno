function image_rec = JPEG_DECOMP(pathFile)
    jpeg = fopen(pathFile,"rb");
    
    % const
    quality_factor = fread(jpeg,1,"uint16");
    og_sz = fread(jpeg,2,"uint16")';
    szIm = fread(jpeg,2,"uint16")';
    % % % LUMINANCE 
    % DC
    profundidad_dc_lum_dict = fread(jpeg,1,"ubit5");
    largo_vector = fread(jpeg,1,"uint32");
    DC_LUM_CODED = fread(jpeg,largo_vector,"ubit1")';
    sz = fread(jpeg,1,"uint16");
    DC_LUM_DICT = cell(sz,2);
    for i = 1:sz
        DC_LUM_DICT{i,1} = fread(jpeg,1,"bit"+string(profundidad_dc_lum_dict));
        large = fread(jpeg,1,"uint8");
        DC_LUM_DICT{i,2} = fread(jpeg,large,"ubit1")';
    end
    % AC
    profundidad_ac_lum_dict = fread(jpeg,1,"ubit5");
    largo_vector = fread(jpeg,1,"uint32");
    AC_LUM_CODED = fread(jpeg,largo_vector,"ubit1")';
    sz = fread(jpeg,1,"uint16");
    AC_LUM_DICT = cell(sz,2);
    for i = 1:sz
        AC_LUM_DICT{i,1} = fread(jpeg,1,"bit"+string(profundidad_ac_lum_dict));
        large = fread(jpeg,1,"uint8");
        AC_LUM_DICT{i,2} = fread(jpeg,large,"ubit1")';
    end
    % % % CROMINANCE
        % DC
    profundidad_dc_crom_dict = fread(jpeg,1,"ubit5");
    largo_vector = fread(jpeg,1,"uint32");
    DC_CROM_CODED = fread(jpeg,largo_vector,"ubit1")';
    sz = fread(jpeg,1,"uint16");
    DC_CROM_DICT = cell(sz,2);
    for i = 1:sz
        DC_CROM_DICT{i,1} = fread(jpeg,1,"bit"+string(profundidad_dc_crom_dict));
        large = fread(jpeg,1,"uint8");
        DC_CROM_DICT{i,2} = fread(jpeg,large,"ubit1")';
    end
    % AC
    profundidad_ac_crom_dict = fread(jpeg,1,"ubit5");
    largo_vector = fread(jpeg,1,"uint32");
    AC_CROM_CODED = fread(jpeg,largo_vector,"ubit1")';
    sz = fread(jpeg,1,"uint16");
    AC_CROM_DICT = cell(sz,2);
    for i = 1:sz
        AC_CROM_DICT{i,1} = fread(jpeg,1,"bit"+string(profundidad_ac_crom_dict));
        large = fread(jpeg,1,"uint8");
        AC_CROM_DICT{i,2} = fread(jpeg,large,"ubit1")';
    end
    fclose(jpeg);
    
    % % % LUM

    AC_LUM = huffmandeco(AC_LUM_CODED,AC_LUM_DICT);
    AC_LUM = runLengthDecoding(AC_LUM);
    DC_LUM = huffmandeco(DC_LUM_CODED,DC_LUM_DICT);
    DC_LUM = cumsum(DC_LUM);
    AC_LUM(1:64:end) = DC_LUM;

    % % % CROM

    AC_CROM = huffmandeco(AC_CROM_CODED,AC_CROM_DICT);
    AC_CROM = runLengthDecoding(AC_CROM);
    DC_CROM = huffmandeco(DC_CROM_CODED,DC_CROM_DICT);
    DC_CROM = cumsum(DC_CROM);
    AC_CROM(1:64:end) = DC_CROM;



    szlum   = szIm;

    szcrom  = szlum/2;
    
    Y_q  = AC_LUM;
    Cb_q = AC_CROM(1:szcrom(1)*szcrom(2));
    Cr_q = AC_CROM((szcrom(1)*szcrom(2))+1:end);


    
    qmatrix_luminance   = qml(quality_factor);
    qmatrix_crominance  = qmc(quality_factor);


    Y_q_r   = jpeg_decuant(Y_q,qmatrix_luminance,szIm) ;
    Cb_q_r  = jpeg_decuant(Cb_q,qmatrix_crominance,szIm/2) ;
    Cr_q_r  = jpeg_decuant(Cr_q,qmatrix_crominance,szIm/2) ;
    
    if quality_factor <= 50
        Cb_q_r = imresize(Cb_q_r,2,"bilinear"); % Upsampling
        Cr_q_r = imresize(Cr_q_r,2,"bilinear"); % Upsampling   
    else
        Cb_q_r = imresize(Cb_q_r,2); % Upsampling
        Cr_q_r = imresize(Cr_q_r,2); % Upsampling   
    end


    image_rec = uint8(cat(3,Y_q_r,Cb_q_r,Cr_q_r));
    if quality_factor <= 50
        image_rec(:,:,1) = imgaussfilt(image_rec(:,:,1), 0.441);
        image_rec(:,:,2) = imgaussfilt(image_rec(:,:,2), 0.861);
        image_rec(:,:,3) = imgaussfilt(image_rec(:,:,3), 0.8010);
    end

    image_rec = ycbcr2rgb(image_rec);
    image_rec = ipad(image_rec,og_sz(1),og_sz(2));
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
%
function datos_originales = runLengthDecoding(rle)
    datos_originales = [];
        for i = 1:2:length(rle)
        valor = rle(i);
        recuento = rle(i+1);
        datos_originales = [datos_originales, repmat(valor, 1, recuento)];
    end
end
%
function image_rec = jpeg_decuant(image_dct_q,qmatrix,sz)   
    image_dct_q = izigzag(image_dct_q,sz);
    decuant = @(bloque) round(bloque.data .*qmatrix);
    image_dct_dq = blockproc(image_dct_q,[8 8], decuant);
    idct = @(bloque) idct2(bloque.data);
    image_rec = blockproc(image_dct_dq, [8 8], idct);
end
function matrix_original = izigzag(matrix_zigzag,sz)
    zigzag_indices = [ 0,  1,  5,  6, 14, 15, 27, 28;
                      2,  4,  7, 13, 16, 26, 29, 42;
                      3,  8, 12, 17, 25, 30, 41, 43;
                      9, 11, 18, 24, 31, 40, 44, 53;
                      10, 19, 23, 32, 39, 45, 52, 54;
                      20, 22, 33, 38, 46, 51, 55, 60;
                      21, 34, 37, 47, 50, 56, 59, 61;
                      35, 36, 48, 49, 57, 58, 62, 63] + 1;

    Cols = size(matrix_zigzag, 2);
    n_matrix = Cols / 64;
    matrix_zigzag = reshape(matrix_zigzag, 8, 8, n_matrix);

    matrix_original = zeros(8, 8, n_matrix);

    for i = 1:n_matrix
        aux = matrix_zigzag(:, :, i);
        matrix_original(:, :, i) = aux(zigzag_indices);
    end
    matrix_original = iSubdividir(matrix_original,sz);
end
function matriz = iSubdividir(submatrices,sz)
    tamanoOriginal = sz;
    matriz = zeros(tamanoOriginal);
    k = 1;
    for i = 1:8:tamanoOriginal(1)
        for j = 1:8:tamanoOriginal(2)
            matriz(i:i+8-1, j:j+8-1) = submatrices(:, :, k);
            k = k + 1;
        end
    end
end
%
function depaddedImage = ipad(image, targetRows, targetCols)
    depaddedImage = image(1:targetRows, 1:targetCols, :);
end