function y = lwlr(X_train, y_train, x, tau)
%%% YOUR CODE HERE
m = size(X_train,1);
n = size(X_train,2);
w = zeros(m,1);
z = zeros(m,1);
h = zeros(m,1);
D = zeros(m,m);
theta = zeros(n,1); 
lamda = 0.0001;
Grad = ones(n,1);
Hess = ones(n,n);
w = exp(-(sum((X_train - x').^2,2)/(2*tau*tau)));
while (norm(Grad) > 1e-6)
      h = 1./(1+exp(-X_train*theta)); 
      z = w.*(y_train-h);
      D = diag(-w.*h.*(1-h));
  for i =1:m
      %D(i,i) = -w(i)*h(i)*(1-h(i));
  end
  Hess = X_train'*D*X_train-lamda*eye(n);
  Grad = X_train'*z-lamda*theta;
  theta = theta - inv(Hess)*Grad;
end

y= double(x'*theta > 0);
