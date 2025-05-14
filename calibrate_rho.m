function [calibrated_rho, loss_value] = calibrate_rho(observed_joint_probs, z_BBB, z_A, mode)
    rho_min = 0;
    rho_max = 1;
    
    objective = @(rho) loss_function(rho, observed_joint_probs, z_BBB, z_A, mode);    
    if strcmp(mode, 'gradient_descent')
        [calibrated_rho, loss_value] = gradient_descent_optimizer(objective, rho_min, rho_max);
    else        
        num_grid_points = 20;
        grid_rhos = linspace(rho_min + 0.01, rho_max - 0.01, num_grid_points);
        grid_losses = zeros(size(grid_rhos));
        
        for i = 1:num_grid_points
            grid_losses(i) = objective(grid_rhos(i));
        end        
        [min_grid_loss, min_idx] = min(grid_losses);
        best_grid_rho = grid_rhos(min_idx);
                
        % Use the grid minimum as starting point for fmincon
        rho0 = best_grid_rho;
        % Run fmincon with the best grid point as starting point
        [calibrated_rho, loss_value] = fmincon(objective, rho0, [], [], [], [], rho_min, rho_max, []);
        
        % Check if fmincon actually improved the result
        if loss_value > min_grid_loss
            warning('fmincon did not improve on the grid search result. Using grid result instead.');
            calibrated_rho = best_grid_rho;
            loss_value = min_grid_loss;
        end
    end
    
    % Final result
    fprintf('Calibrated rho: %.6f with loss: %.8e\n', calibrated_rho, loss_value);
    

end