data = load('data.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);

X = [ones(m, 1), data(:,1)];
theta = zeros(2, 1);

iterations = 1000;
alpha = 0.01;

J = computeCost(X, y, theta);

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, J_history] = gradientDescent(@computeCost, X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

figure;
plot(X(:,2), X*theta, '-');
hold on
scatter(X(:, 2), y);
hold off

figure;
plot(1:iterations, J_history, '-');
