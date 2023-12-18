function COLORCOMP(imagen_original,k)
    % Parametros Iniciales %
    ITERACIONES = 10;
    og_size = size(imagen_original);
    imagen_original_vectorizada = single(reshape(imagen_original,[],3));
    % -------------------- %
    % Asignacion de peso para colores unicos %
    [colores_unicos,~,indices_CU] = unique(imagen_original_vectorizada,"rows");
    rep = histcounts(indices_CU,length(colores_unicos))';    
    colores_con_rep = rep.*colores_unicos;
    % -------------------------------------- %
    % Inicializacion de pesos a partir de K-means ++ %
    if k*6 <= length(colores_unicos)
        x = k*6;
    elseif k <=length(colores_unicos)
        x = k;
    else
        x = length(colores_unicos);
    end
    submuestra = colores_unicos(randperm(size(colores_unicos, 1),x),:);
    sz = size(submuestra);
    primer_color = randi([1, sz(1)], 1, 1, 'single');
    colores(1, :) = submuestra(primer_color, :);
    for i = 2:k
        distancias = min(pdist2(submuestra, single(colores(1:i-1,:))),[],2);
        probabilidades = distancias / sum(distancias);
        siguiente_color = randsample(1:length(probabilidades), 1, true, probabilidades);
        colores(i, :) = submuestra(siguiente_color, :);
    end
    %------------------------------------------------%
    % Algotitmo K-means %
    for iteracion = 1:ITERACIONES
        minima_distancia = pdist2(colores_unicos, single(colores));
        [~, indices] = min(minima_distancia, [], 2);
        if iteracion ~= ITERACIONES
            for color = 1:k
                    color_indices = find(indices == color);
                    if isempty(color_indices) && ITERACIONES == iteracion
                        colores(color,:) = [];
                    elseif ~isempty(color_indices)
                        colores(color, :) = sum(colores_con_rep(color_indices, :))/sum(rep(color_indices));
                    else
                        random_index = randi([1, length(colores_unicos)]);
                        colores(color, :) = colores_unicos(random_index, :);
                    end
            end
        end
    end
    % ---------------------------------------------- %
    % Return de datos comprimidos %
    IMG_COMP = reshape(indices((indices_CU),:),og_size(1),og_size(2))-1;
    IMG_IDX = uint8(colores);
    IMG_CODED  = [IMG_COMP(:); IMG_IDX(:)];
    % --------------------------- %
    sz_comp = size(IMG_COMP);
    sz_idx = size(IMG_IDX);

    cc = fopen("Image.colorComp","wb");
    fwrite(cc,sz_comp,"single");
    fwrite(cc,sz_idx,"single");
    arit(IMG_CODED,cc);
    fclose(cc);
end