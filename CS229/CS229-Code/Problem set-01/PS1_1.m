clear
% load data and run  
X = load('logistic_x.txt');
Y = load('logistic_y.txt');
[a1,V1]=find(Y > 0);
[a2,V2]=find(Y < 0);
max_i = 1000;
X = [ones(size(X, 1), 1) X];
[Theta, J] = NT(X, Y, max_i );
[Theta_GD, J_GD] = GD(X, Y, 0.5 , max_i);
[Theta_SGD, J_SGD] = SGD(X, Y, 0.5 , max_i);
figure; 
hold on;
plot(1:max_i, J);
plot(1:max_i, J_GD,'r' );
plot(1:max_i, J_SGD,'k' );
%% plot logistic regression result
m = size(X,1);
figure; 
hold on;
%plot(X(a1, 2 ), X(a1, 3), 'rx' );
%plot(X(a2, 2 ), X(a2, 3), 'bo' );
plot( X( Y < 0, 2 ), X( Y < 0, 3), 'rx' );
plot( X( Y > 0, 2 ), X( Y > 0, 3), 'bo' );

% line
x1 = min(X(:,2)) : 0.01 : max(X(:,2));
x2 = -(Theta(1) / Theta(3)) - (Theta(2) / Theta(3)) * x1;
plot(x1,x2,'b');

x1_GD = min(X(:,2)) : 0.01 : max(X(:,2));
x2_GD = -(Theta_GD(1) / Theta_GD(3)) - (Theta_GD(2) / Theta_GD(3)) * x1;
plot(x1_GD,x2_GD,'r');

x1_SGD = min(X(:,2)) : 0.01 : max(X(:,2));
x2_SGD = -(Theta_SGD(1) / Theta_SGD(3)) - (Theta_SGD(2) / Theta_SGD(3)) * x1;
plot(x1_SGD,x2_SGD,'k');
xlabel('x1');
ylabel('x2');

%% predic accurancy
%Newton
z = X * Theta; % !!!
g_z = 1 ./ (1 + exp(-z));
g_z = [g_z >= 0.5];
%Gradient D
z_GD = X * Theta_GD; % !!!
g_z_GD = 1 ./ (1 + exp(-z_GD));
g_z_GD = [g_z_GD >= 0.5];
%SGD
z_SGD = X * Theta_SGD; % !!!
g_z_SGD = 1 ./ (1 + exp(-z_SGD));
g_z_SGD = [g_z_SGD >= 0.5];
Y = [Y > 0];
fprintf('accurancy:%f\n', sum(Y==g_z)/size(Y, 1));
fprintf('accurancy GD:%f\n', sum(Y==g_z_GD)/size(Y, 1));
fprintf('accurancy SGD:%f\n', sum(Y==g_z_SGD)/size(Y, 1));