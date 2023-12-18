
function l = laplacian_pyramid(imagen,N,a) 

    g = cell(1,N+1);
    e = cell(1,N+1);
    l = cell(1,N);
    
    g{1} = imagen;
    for i = 2:N
        g{i} = REDUCE(g{i-1},a);
    end
    
    e{N} = g{N};
    for i = N-1:-1:1
       e{i} = EXPAND(e{i+1},a);
    end
    
    l{1} = double(g{1}) - double(EXPAND(g{2},a));
    for i = 2:N-1
        l{i} = double(g{i}) - double(EXPAND(g{i+1},a));
    end
    l{N} = g{N};
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

function  Im_g = REDUCE(Im_anterior,a)
    w = [1/4-a/2 1/4 a 1/4 1/4-a/2];
    w2 = w'*w;

    [~,~, channels] = size(Im_anterior);
    
    Im_tmp = imfilter(Im_anterior,w2,"full");
    Im_g = uint8(zeros(size(Im_tmp(3:2:end-2,3:2:end-2,:))));
    for channel = 1:channels
        Im_g(:,:,channel) = uint8(Im_tmp(3:2:end-2,3:2:end-2,channel));
    end
end