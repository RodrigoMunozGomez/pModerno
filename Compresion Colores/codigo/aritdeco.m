function vector = aritdeco(wtr)
    largo = fread(wtr,1,"single");
    aritCoded = fread(wtr,largo,"ubit1");
    largoFrecUnique = fread(wtr,1,"single");
    aritUnique = fread(wtr,largoFrecUnique,"single");
    aritFrec = fread(wtr,largoFrecUnique,"single");
    aritLen = fread(wtr,1,"single");
    vector = arithdeco(aritCoded,aritFrec,aritLen);
    vector = aritUnique(vector,:)';
end