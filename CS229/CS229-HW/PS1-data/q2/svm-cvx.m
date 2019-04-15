load data/x.dat
load data/y.dat
X = x;
Y = 2*(y-0.5);%change -1,1 into 1,0

C = 1;
m = size(X,1);
n = size(X,2);

cvx_begin
    variable w(n) b xi(m);
    minimize 1/2*sum(w.*w) + C*sum(xi);
    y.*(X*w+b) >= 1-xi;
    xi >= 0
cvx_end

xp = linspace(min(X(:,1))), max(X(:,1)),100);
yp = -(w(1)*xp + b)/w(2);
yp1 = -(w(1)*xp + b)/w(2);
