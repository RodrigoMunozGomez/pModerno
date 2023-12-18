
function imagen_recon = laplacian_pyramid_recon(l,a)
    
    [~,N] = size(l);
    d = cell(1,N-1);
    d{N-1} = uint8( double(EXPAND(l{N},a)) + double(l{N-1}) ) ;
    for i = N-2:-1:1
        d{i} = uint8( double(EXPAND(d{i+1},a)) + double(l{i}) ) ;
    end
    imagen_recon = d{1};
end
function Im_e = EXPAND(Im_g,a)

    w = [1/4-a/2 1/4 a 1/4 1/4-a/2];
    w2 = w'*w;
    [R,C,channels] = size(Im_g);
    Im_tmp = uint8( zeros([2*[R C] - 1 ,channels] ) );
    Im_e = Im_tmp;
    for channel = 1:channels
        Im_tmp(1:2:end,1:2:end,channel) = Im_g(:,:,channel);
        Im_e(:,:,channel) = imfilter(Im_tmp(:,:,channel),4*w2,"same");
    end
end