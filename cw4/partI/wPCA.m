% to be completed

function U_reduc = wPCA(trainset, bcc)
    X = trainset;
    avg = mean(X, 1);
    X = X - repmat(avg, size(X, 1), 1);
    sigma = X * X' / size(X, 2);
    [U,S,V] = svd(sigma);
    xPCAwhite = diag(1./sqrt(diag(S) + 1e-5)) * U' * X;
    U_reduc = transpose(xPCAwhite);
end
