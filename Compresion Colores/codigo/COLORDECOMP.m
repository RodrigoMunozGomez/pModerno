function [IMG_RECON] = COLORDECOMP 
    cc = fopen("Image.colorComp","rb");
    size_comp = fread(cc,2,"single")';
    size_idx = fread(cc,2,"single")';
    totalCode = aritdeco(cc);
    fclose(cc);
    lengthCode = size_comp(1)*size_comp(2);
    IMG_COMP = reshape( totalCode(1,1:lengthCode),size_comp);
    IMG_IDX  = reshape( totalCode(1,lengthCode+1:end),size_idx);
    
    
    
    sz = size_comp;
    IMG_COMP = reshape(IMG_COMP+1,[],1);
    IMG_RECON = IMG_IDX(IMG_COMP,:);
    IMG_RECON = uint8(reshape(IMG_RECON,[sz 3]));
end