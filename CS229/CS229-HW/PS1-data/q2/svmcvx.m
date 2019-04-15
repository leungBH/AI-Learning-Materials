load data/x.dat
load data/y.dat
X = x;
Y = 2*(y-0.5);%change -1,1 into 1,0

C = 1;
m = size(X,1);
n = size(X,2);

cvx_begin
    variables w(n) b xi(m);
    minimize 1/2*sum(w.*w) + C*sum(xi);
    Y.*(X*w+b) >= 1-xi;
    xi >= 0
cvx_end

xp = linspace(min(X(:,1)), max(X(:,1)),100);
yp = -(w(1)*xp + b)/w(2);
yp1 = -(w(1)*xp + b-1)/w(2);
yp0 = -(w(1)*xp + b+1)/w(2);
idx0 = find(y ==0);
idx1 = find(y ==1);
plot(x(idx0,1),x(idx0,2),'rx');hold on 
plot(x(idx1,1),x(idx1,2),'go');
plot(xp,yp,'-b',xp,yp1,'--g',xp,yp0,'--r');
hold off





