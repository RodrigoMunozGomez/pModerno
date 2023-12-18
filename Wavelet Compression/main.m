Imagen = imread("lena.tif");

Imagen_shifted = double(Imagen) - 2^7;

Imagen_ycbcr = ycbcr(Imagen_shifted);

Y = Imagen_ycbcr(:,:,1);
Cb = Imagen_ycbcr(:,:,2);
Cr = Imagen_ycbcr(:,:,3);



levels   =  4    ;
stepSize =  16   ; % 8 - 6 - 4
decimacion = 1;
% if decimacion == 1
%     Cb = imresize(Cb,0.5);
%     Cr = imresize(Cr,0.5);
% end


[bit_stream_Y, S_Y] = wavelet_custom(Y,levels,stepSize);
[bit_stream_Cb, S_Cb] = wavelet_custom(Cb,levels,stepSize);
[bit_stream_Cr, S_Cr] = wavelet_custom(Cr,levels,stepSize);

bit_stream_coded = [bit_stream_Y,bit_stream_Cb,bit_stream_Cr];
% [Bit_stream_coded,Bit_stream_dict] = huff(bit_stream_coded);
[aritBit,aritUnicos, aritFrec, aritLen] = arit(bit_stream_coded);
first = aritUnicos(1,1);
aritUnicos = diff(aritUnicos);
prof = ceil(log2(max(aritUnicos)))+1;

comp = fopen("Imagen.Comp","wb");
fwrite(comp,stepSize,"ubit10");
fwrite(comp,levels,"ubit4");
fwrite(comp,S_Y,"single");
fwrite(comp,S_Cb,"single");

fwrite(comp,length(bit_stream_Y),"single");
fwrite(comp,length(bit_stream_Cb),"single");

fwrite(comp,length(aritBit),"single");

fwrite(comp,aritBit,"ubit1");
fwrite(comp,length(aritFrec),"single");
fwrite(comp,aritFrec,"single");
fwrite(comp,aritLen,"single");
fwrite(comp,length(aritUnicos),"single");
fwrite(comp,prof,"ubit5");
fwrite(comp,aritUnicos,"ubit"+string(prof));
fwrite(comp,first,"single");
fclose(comp);



