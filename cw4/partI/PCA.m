% to be completed

function U_reduc = PCA(trainset, bcc)
    % trainset: 340x4096 matrix
    % bias_correction_constant = N-1 = 339

    X = trainset
    X = bsxfun(@minus, X, mean(X,1));
    C = (X'*X)./bcc; 

    [V D] = eig(C);
    [D order] = sort(diag(D), 'descend');
    V = V(:,order);

    U_reduc_raw = X*V(:,1:end);
    U_reduc = transpose(U_reduc_raw)
end

