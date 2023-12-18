function [bit_stream,S] = wavelet_custom(matrix, level, stepSize)
    dwtmode('per','nodisp')
    waveletName = "bior4.4";
    [C, S] = wavedec2(matrix,level,waveletName,'per');
    CodeBlocksPerSuband = cell(3,level);
    bit_stream = reshape(dzquant(appcoef2(C, S, waveletName),stepSize),[],1)';
    for i = level:-1:1
        H = reshape(dzquant(detcoef2('h', C, S, i),stepSize),[],1)';
        V = reshape(dzquant(detcoef2('v', C, S, i),stepSize),[],1)';
        D = reshape(dzquant(detcoef2('d', C, S, i),stepSize),[],1)';

        bit_stream = [bit_stream, H, V, D];
    end
end


function q = dzquant(vector,step)
    q = sign(vector).*floor(abs(vector)./step);
end

