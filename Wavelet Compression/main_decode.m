comp = fopen("Imagen.Comp","rb");

stepSize = fread(comp,1,"ubit10");
levels = fread(comp,1,"ubit4");
S_Lum = fread(comp,(levels + 2)*2,"single");
S_Lum = reshape(S_Lum,[],2);

S_crom = fread(comp,(levels + 2)*2,"single");
S_crom = reshape(S_crom,[],2);


largo_lum = fread(comp,1,"single");
largo_crom = fread(comp,1,"single");

largo_coded = fread(comp,1,"single");
arit_code = fread(comp,largo_coded,"ubit1");
largo_frec = fread(comp,1,"single");
arit_frec = fread(comp,largo_frec,"single");
arit_Len = fread(comp,1,"single");
largo_Unicos = fread(comp,1,"single");
prof = fread(comp,1,"ubit5");
arit_Unicos = fread(comp,largo_Unicos,"ubit"+string(prof));

first = fread(comp,1,"single");
arit_Unicos = [first;arit_Unicos];
arit_Unicos = cumsum(arit_Unicos);

fclose(comp);


stream = arithdeco(arit_code,arit_frec,arit_Len);


stream = arit_Unicos(stream,:);

streamY  = stream(1:largo_lum);
index = largo_lum;
streamCb = stream(index + 1 :index + largo_crom); 
index = index + largo_crom;
streamCr = stream(index + 1:index + largo_crom);

streamY = dzdequant(streamY,stepSize);
streamCb = dzdequant(streamCb,stepSize);
streamCr = dzdequant(streamCr,stepSize);

Y = waverec2(streamY,S_Lum,"bior4.4");
Cb = waverec2(streamCb,S_crom,"bior4.4");
Cr = waverec2(streamCr,S_crom,"bior4.4");


Im_rec = cat(3,Y,Cb,Cr);
Im_rec = iycbcr(Im_rec);
Im_rec = uint8(Im_rec + 2^7);

figure color w

im = imread("lena.tif");
subplot(1,2,1)

imshow(im) ; title("Imagen Original");

subplot(1,2,2)

imshow(Im_rec) ; title("Imagen Reconstruida")

xlabel(sprintf("PSNR : %.2f",psnr(Im_rec,im)))
