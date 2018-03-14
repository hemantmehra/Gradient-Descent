function [theta, J_history] = gradientDescent(J, X, y, theta, alpha, num_iters)
    J_history = zeros(num_iters, 1);
    m = length(y);
    n = length(theta);
    delta = zeros(n, 1);
    epsilon = 0.001;
    
    for iter = 1:num_iters
        for j = 1:n
            temp1 = theta;
            temp2 = theta;
            temp1(j) = temp1(j) + epsilon;
            temp2(j) = temp2(j) - epsilon;
            delta(j) = (J(X, y, temp1) - J(X, y, temp2))/(2 * epsilon);
        end
        
        theta = theta - alpha * delta;
        J_history(iter) = J(X, y, theta);
    end
end