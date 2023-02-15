function MatrixGen()
    for n=5:5:1000
        A=zeros(n,5);
        A(1,1)=n*5;
        dlmwrite(['Matrices/' int2str(n) '.txt'], A, 'Delimiter', ' ', 'precision','%.6f')
    end
end