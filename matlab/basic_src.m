function [accu, co] = basic_src(tr)
cls = 38;
[sam_tr, ~, sam_te, ind_te] = predat(tr);
dim = size(sam_tr, 1);
numOte = size(sam_te, 2);
p_value = zeros(1, numOte);
parfor sam = 1:numOte
    sam
    opts = spgSetParms('verbosity',0);         % Turn off the SPGL1 log output
    x = spg_bp(sam_tr, sam_te(:, sam), opts);
    res = zeros(1, 38);
    sam_tr_r = reshape(sam_tr, [dim tr cls]);
    x_r = reshape(x, [tr cls]);
    for i = 1:38
        res(i) = norm(sam_te(:, sam)-sam_tr_r(:, :, i)*x_r(:, i), 2);
    end
    [~, ind] = min(res);
    p_value(sam) = ind;
end
co = confusionmat(ind_te, p_value);
accu = trace(co)/sum(sum(co));