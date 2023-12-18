function arit(vector,wtr) 
    aritLen = length(vector);
    [aritUniques, ~, idx] = unique(vector);
    aritFrec = histc(vector, aritUniques);
    vector_indices = idx;
    aritCoded = arithenco(vector_indices, aritFrec);
    fwrite(wtr,length(aritCoded),"single");
    fwrite(wtr,aritCoded,"ubit1");
    fwrite(wtr,length(aritFrec),"single");
    fwrite(wtr,aritUniques,"single");
    fwrite(wtr,aritFrec,"single");
    fwrite(wtr,aritLen,"single");
end
