%reading in the trainig data used to learn the datatransformation

m = readtable('Out\train.csv');
m = m(2:end,2:end);
m.Label = m.Var302;

XTr = table2array(m(:,1:300));
XTr = XTr.';
YTr = m.Label;

filename = 'Out\test.csv';
if isfile(filename)
    n = readtable('Out\test.csv');
    n = n(2:end,2:end);
    n.Label = n.Var302;
    XT = table2array(n(:,1:300));
    XT = XT.';
end   

%%
%Training the model
params = struct();
params.kernel = 0;
params.knn = 5;
params.k1 = 5;
params.k2 = 5;
params.dim = 2;
L = DMLMJ(XTr, YTr, params);

%%
%Transforming the new data.

Tr = (L'*XTr)';
Tr = array2table(Tr);
Tr.Label = m.Label;
writetable(Tr,'In\train_in.csv');

if isfile(filename)
    T = (L'*XT)';
    T = array2table(T);
    T.Label = n.Label;
    writetable(T,'In\test_in.csv');
end


