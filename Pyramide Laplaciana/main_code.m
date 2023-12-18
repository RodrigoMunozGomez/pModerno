imagen = imread("lenna.tif");
imagen = imresize(imagen,[513 513]);
[Rows,Cols,Channels] = size(imagen);

w_factor    = 0.55   ;
Niveles     = 6     ;
profundidad = [1,2,3]; %% Los dos ultimos niveles son insignificantes en cuanto a memoria, por lo cual se opta por no tocarlos

piramide = laplacian_pyramid(imagen,Niveles,w_factor);

[piramide_cuant, C] = kcuant(piramide,profundidad); %% Guardar C

% SAVE
%%
lap = fopen("comp_final.ror","wb");
fwrite(lap,w_factor,"single");
fwrite(lap,Niveles,"ubit3");
fwrite(lap,profundidad,"ubit4");
fwrite(lap,[Rows, Cols],"ubit16");
fwrite(lap,Channels,"ubit2");
for i = 1:3
    fwrite(lap,C{1,i},"int8");
end

fwrite(lap,piramide_cuant{4},"int8");
fwrite(lap,piramide_cuant{5},"int8"); %% Ultimos Niveles 
fwrite(lap,piramide_cuant{6},"uint8");


l_vector = [piramide_cuant{1}(:)' , piramide_cuant{2}(:)', piramide_cuant{3}(:)'];
l_vector = runLengthEncoding(l_vector);
[l_vector_codificado, dict_l_vector] = huff(l_vector);

fwrite(lap,length(l_vector_codificado),"single");
fwrite(lap,l_vector_codificado,"ubit1");

sz = size(dict_l_vector);
fwrite(lap,sz(1),"single");

for i = 1:sz(1)
    fwrite(lap,dict_l_vector{i,1},"single");
    fwrite(lap,length(dict_l_vector{i,2}), "single");
    fwrite(lap,dict_l_vector{i,2},"ubit1");
end
fclose(lap);

%% LOAD
clc; clear;
lap = fopen("comp_final.ror","rb");
w_factor = fread(lap,1,"single");
Niveles = fread(lap,1,"ubit3");
Profundidad = fread(lap,Niveles-3,"ubit4")';
sz = fread(lap,2,"ubit16");
all_sz = zeros(Niveles,2);
all_sz(1,:) = sz ;
for i = 2:Niveles
    all_sz(i,1) = round(all_sz(i-1,1)/2);
    all_sz(i,2) = round(all_sz(i-1,2)/2);
end
Channels = fread(lap,1,"ubit2");
C = cell(1,3);
for i = 1:3
   C{i} =  fread(lap,(2^Profundidad(i))*Channels,"int8");
   C{i} = reshape(C{i},Channels,2^Profundidad(i));
end
piramide_decod = cell(1,Niveles);

piramide_decod{4} = fread(lap,prod(all_sz(4,:))*Channels,"int8");
piramide_decod{4} = reshape(piramide_decod{4},all_sz(4,1),all_sz(4,2),Channels);
piramide_decod{5} = fread(lap,prod(all_sz(5,:))*Channels,"int8"); 
piramide_decod{5} = reshape(piramide_decod{5},all_sz(5,1),all_sz(5,2),Channels);
piramide_decod{6} = fread(lap,prod(all_sz(6,:))*Channels,"uint8");
piramide_decod{6} = reshape(piramide_decod{6},all_sz(6,1),all_sz(6,2),Channels);

large_l = fread(lap,1,"single");
l_vec_cod = fread(lap,large_l,"ubit1")';
sz = fread(lap,1,"single");
dict_l_vector = cell(sz,2);

for i = 1:sz
    dict_l_vector{i,1} = fread(lap,1,"single");
    large = fread(lap,1,"single");
    dict_l_vector{i,2} = fread(lap,large,"ubit1")';
end
l_vec_decod = huffmandeco(l_vec_cod,dict_l_vector);
l_vec_decod = runLengthDecoding(l_vec_decod);
fclose(lap);
start_index = 1;
for N = 1:Niveles-3
    end_index = start_index + (all_sz(N,1)*all_sz(N,2)*Channels) - 1;
    piramide_decod{N} = l_vec_decod(start_index:end_index);
    piramide_decod{N} = reshape(piramide_decod{N},all_sz(N,1),all_sz(N,2),Channels);
    start_index = end_index + 1;
end
%% Decuant

[piramide_decod] = kdecuant(piramide_decod,C,Profundidad);
imagen_recon = laplacian_pyramid_recon(piramide_decod,w_factor);
imagen = imread("lenna.tif");
imagen = imresize(imagen,[513 513]);

subplot(1,2,1)
imshow(imagen)
subplot(1,2,2) 
imshow(imagen_recon)
xlabel(sprintf("PSNR : %.2f",psnr(imagen_recon,imagen)))

%% Funciones
function [vector_codificado, huffman_dict] = huff(piramide_cuant)
    frecuencia = histc(piramide_cuant,unique(piramide_cuant));
    probabilidad = frecuencia/length(piramide_cuant);
    huffman_dict = huffmandict(unique(piramide_cuant), probabilidad);
    vector_codificado = huffmanenco(piramide_cuant, huffman_dict);
end
function [piramide_kcuant, C_new] = kcuant(piramide,profundidad)
    C_new = cell(1,length(profundidad));
    piramide_kcuant = piramide;
    for Nivel = 1:length(profundidad)
        [~,~,channels] = size(piramide{Nivel});
        sz = size(piramide{Nivel});
        for channel = 1:channels
            piramide_channel = piramide{Nivel}(:,:,channel);
            [ aux , C ] = kmeans( reshape(piramide_channel,[],1) , 2^profundidad(Nivel) , "Replicates" , 15);
            aux_1 = zeros(size(aux));
            % Calcula la frecuencia de cada valor único en aux
            unique_values = unique(aux);
            value_frequencies = histc(aux, unique_values);
            
            % Ordena C en función de las frecuencias en orden descendente
            [~, sorted_indices] = sort(value_frequencies, 'descend');
            sorted_C = C(sorted_indices, :);
            
            % Asigna los valores de sorted_C a aux
            for i = 1:length(unique_values)
                aux_1(aux == sorted_indices(i)) = i;
            end
            aux = aux_1;
            C_new{Nivel}(channel,:) = round(sorted_C);
            aux = reshape(aux,sz(1),sz(2));
            piramide_kcuant{Nivel}(:,:,channel) = aux-1;
        end

    end
end
function [piramide_decuant] = kdecuant(piramide_cuant, C, profundidad)
    piramide_decuant = piramide_cuant;
    for Nivel = 1:length(profundidad)
        [~,~,channels] = size(piramide_cuant{Nivel});
        sz = size(piramide_cuant{Nivel});
        aux = zeros(sz);
        for channel = 1:channels
            aux_channel = aux(:,:,channel);
            for i = 1:2^profundidad(Nivel)
                  aux_channel(piramide_cuant{Nivel}(:,:,channel) == i-1) = C{Nivel}(channel,i);
                  aux(:,:,channel) = aux_channel;
            end
        end
        piramide_decuant{Nivel} = aux;
    end
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
function datos_originales = runLengthDecoding(rle)
    datos_originales = [];
    for i = 1:2:length(rle)
        valor = rle(i);
        recuento = rle(i+1);
        datos_originales = [datos_originales, repmat(valor, 1, recuento)];
    end
end
