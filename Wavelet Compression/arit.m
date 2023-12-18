function [vector_coded,valores_unicos, frecuencia, largo] = arit(vector)
    % Calcula la longitud del vector
    largo = length(vector);

    % Encuentra los valores únicos y sus frecuencias
    [valores_unicos, ~, idx] = unique(vector);
    frecuencia = histc(vector, valores_unicos);

    % Convierte el vector original a índices basados en los valores únicos
    % La función unique devuelve también un vector de índices (idx) que mapea
    % cada elemento del vector original a su posición en valores_unicos
    vector_indices = idx;

    % Realiza la codificación aritmética utilizando los índices y las frecuencias
    vector_coded = arithenco(vector_indices, frecuencia);
end
