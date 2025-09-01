function [rho_opt, loss_opt] = gradient_descent_optimizer(objective, rho_min, rho_max)
    % Implementation of gradient descent optimization for a bounded 1D problem
    
    % Gradient descent parameters
    learning_rate = 0.01;      % Step size
    max_iterations = 1000;     % Maximum number of iterations
    convergence_tol = 1e-8;    % Convergence tolerance
    
    % Multiple starting points to avoid local minima
    num_starts = 10;
    starting_points = linspace(rho_min + 0.01, rho_max - 0.01, num_starts);
    
    % Initialize arrays to store results from each starting point
    best_rhos = zeros(1, num_starts);
    best_losses = zeros(1, num_starts);
    
    % Try each starting point
    for i = 1:num_starts
        start_rho = starting_points(i);
        rho = start_rho;
        prev_loss = inf;        
        % Main gradient descent loop
        for iter = 1:max_iterations
            current_loss = objective(rho);            
            % Calculate gradient using finite differences
            h = 1e-5;  % Small step for numerical derivative
            forward_loss = objective(min(rho + h, rho_max));
            backward_loss = objective(max(rho - h, rho_min));
            gradient = (forward_loss - backward_loss) / (2 * h);            
            new_rho = rho - learning_rate * gradient;
            
%            if abs(new_rho - rho) < convergence_tol || abs(current_loss - prev_loss) < convergence_tol
 %               fprintf('Converged after %d iterations\n', iter);
  %              break;
   %         end
            
            % Update for next iteration
            rho = new_rho;
            prev_loss = current_loss;
        end        
        final_loss = objective(rho);
        
        % Store result for this starting point
        best_rhos(i) = rho;
        best_losses(i) = final_loss;
        
    %    fprintf('Final rho: %.6f with loss: %.8e\n\n', rho, final_loss);
    end
    
    % Find the best result across all starting points
    [loss_opt, idx] = min(best_losses);
    rho_opt = best_rhos(idx);
    
    fprintf('Best result: rho = %.6f with loss = %.8e\n', rho_opt, loss_opt);
end