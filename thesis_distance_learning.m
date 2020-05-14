%reading in the trainig data used to learn the datatransformation

m = readtable('Out\train.csv');
m = m(2:end,2:end);
Label_train = table2array(m(:,end));

XTr = table2array(m(:,1:end-1));
XTr = XTr.';
YTr = table2array(m(:,end));


filename = 'Out\test.csv';
if isfile(filename)
    n = readtable('Out\test.csv');
    n = n(2:end,2:end);
    Label_test = table2array(n(:,end));
    XT = table2array(n(:,1:end-1));
    XT = XT.';
end   

%%
%Training the model
params = struct();
params.kernel = 0;
params.knn = 5;
params.k1 = 5;
params.k2 = 5;
params.dim = dim;
L = DMLMJ(XTr, YTr, params);

%%
%Transforming the new data.

Tr = (L'*XTr)';
Tr = array2table(Tr);
Tr.Label = Label_train;
writetable(Tr,'In\train_in.csv');

if isfile(filename)
    T = (L'*XT)';
    T = array2table(T);
    T.Label = Label_test;
    writetable(T,'In\test_in.csv');
end
