function [theta, J] = NT(x, y, max_iters)
%NT 此处显示有关此函数的摘要
%   此处显示详细说明
    m = size(x, 1); % sample num
    n = size(x, 2); % feature size
    theta = zeros(n, 1);
    J = zeros(max_iters, 1);

    for ti = 1 : max_iters
        z = x * theta .* y;
        J(ti) = 1 / m * sum( log(1 + exp(-z)) );
        g_z = 1 ./ (1 + exp(-z));

        grad = (1 / m) * x' * ( (g_z - 1) .* y );
        hession = (1 / m) * x' * diag( g_z .* (1 - g_z) ) * x;    % y * y = 1

        theta = theta - inv(hession) * grad;
    end


end

