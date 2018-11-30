function [] = partII()

    % generate the data

    rng(1); 
    r = sqrt(rand(100,1)); 
    t = 2*pi*rand(100,1);  
    data1 = [r.*cos(t), r.*sin(t)]; 

    r2 = sqrt(3*rand(100,1)+1); 
    t2 = 2*pi*rand(100,1);      
    data2 = [r2.*cos(t2), r2.*sin(t2)]; 

    % plot the data

    figure;
    plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
    hold on
    plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
    axis equal
    hold on

    % work on class 1
    [a1, R1] = calcRandCentre(data1);

    % work on class 2
    [a2, R2] = calcRandCentre(data2);

    % plot centre and radius for class 1
    plot(a1(1), a1(2), 'rx', 'MarkerSize', 15);
    viscircles(a1', R1, 'Color', 'r', 'LineWidth', 1);
    hold on

    % plot centre and radius for class 2
    plot(a2(1), a2(2), 'bx', 'MarkerSize', 15);
    viscircles(a2', R2, 'Color', 'b', 'LineWidth', 1);

end

function k = kernel(x_i, x_j, w)
    %k = exp(-norm(x_i - x_j,2)^2/w);
    % non-linear space kernel
    k = x_i'*x_j/w;
end

function [a, R] = calcRandCentre(data)

    n = size(data, 1);
    H = 2 * data * data';
    gl = zeros(n, 1);
    gu = ones(n , 1);

    % calculate quads
    quads = quadprog(H, -diag(H), zeros(1, n), 0, ones(1, n), 1, gl, gu);

    % calculate a
    a = zeros(2, 1);
    for i = 1:n
        a = a + data(i, :)' * quads(i);
    end

    % calculate R
    R = 0;
    quad_vec = find(quads>1e-10);
    n = length(quad_vec);
    for i = 1:n
        j = quad_vec(i);
        R = R + norm(data(j, :)' + a);
    end
    R = R/n;

    % disp values
    disp(a)
    disp(R)
end
