function [sam_tr, ind_tr, sam_te, ind_te] = predat(tr)

load('../dataset/Yale.mat');
ind_tr = zeros(38, tr);
for i = 1:38
    idx = find(gnd == i);
    n = length(idx);
    rd = randperm(n, tr);
    ind_tr(i, :) = idx(rd);
end
ind_tr = reshape(ind_tr', [tr*38 1]);
sam_tr = fea(ind_tr, :)';
fea(ind_tr, :) = [];
sam_te = fea';
gnd(ind_tr) = [];
ind_te = gnd;
 
