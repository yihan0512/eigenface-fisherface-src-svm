train = [10 20 30 40 50];
expr_time = 10;
res = zeros(1, 5);
for i = 1:length(train)
    tr = train(i);
    ac = 0;
    for j = 1:expr_time
        [a, co] = basic_src(tr);
        ac = ac + a;
    end
    accu = ac/expr_time;
    res(i) = accu;
end
save('dataset/src_result', 'accu')
