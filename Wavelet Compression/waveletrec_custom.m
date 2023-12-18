function [Im_rec] = waveletrec_custom(LL,CodeBlocksPerSuband,S, stepSize)
    coef = dzdequant((LL,S(1,:)),stepSize,1,1,"HH");
    Coef = reshape(LL,1,[]); sz_index = 2;
    for i = size(S,1)-2 : -1 : 1
        A = reshape(dzdequant(iSubdividir(CodeBlocksPerSuband{1,i},S(sz_index,:)),stepSize,1,i,"HL"),1,[]);
        B = reshape(dzdequant(iSubdividir(CodeBlocksPerSuband{2,i},S(sz_index,:)),stepSize,1,i,"LH"),1,[]);
        C = reshape(dzdequant(iSubdividir(CodeBlocksPerSuband{3,i},S(sz_index,:)),stepSize,1,i,"HH"),1,[]); sz_index = sz_index + 1 ;
        Coef = [Coef, A, B, C];
    end
    Im_rec = waverec2(Coef,S,"bior4.4");
end
    