function  [theta, J] = SGD(x, y, alpha, max_iters)
%GD 此处显示有关此函数的摘要
%   此处显示详细说明
    m = size(x, 1); % sample num
    n = size(x, 2); % feature size
    theta = zeros(n, 1);
    J = zeros(max_iters, 1);
    for ti = 1 : max_iters 
        ran_num = randperm(m,1)
        z = x(ran_num,:) * theta.*y(ran_num);
        g = 1./(1+exp(-z));
        J(ti) = 1 / m * sum( log(1 + exp(-z)) );      
        theta = theta - alpha *  x(ran_num,:)' * ((g-1) .*y(ran_num));         
    end

end

