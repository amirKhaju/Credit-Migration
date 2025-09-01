function [calibrated_rho, loss_value] = calibrate_rho(observed_joint_probs, z_BBB, z_A, mode)
    rho_min = 0;
    rho_max = 1;
    objective = @(rho) loss_function(rho, observed_joint_probs, z_BBB, z_A, mode);    
           
    num_grid_points = 40;
    grid_rhos = linspace(rho_min + 0.01, rho_max - 0.01, num_grid_points);
    grid_losses = zeros(size(grid_rhos));
    
    for i = 1:num_grid_points
        grid_losses(i) = objective(grid_rhos(i));
    end        
    [min_grid_loss, min_idx] = min(grid_losses);
    best_grid_rho = grid_rhos(min_idx);
            
    % Use the grid minimum as starting point for fmincon
    rho0 = best_grid_rho;

    % fmincon options
    opts = optimoptions('fmincon', 'Display','none', 'TolX', 1e-8, 'TolFun', 1e-10);

    % Run fmincon using the best grid point as initial guess
    [calibrated_rho, loss_value] = fmincon( ...
        objective, rho0, ...     % Objective function and initial rho
        [], [], [], [], ...      % No linear constraints
        rho_min, rho_max, ...    % Lower and upper bounds on rho
        [], opts);               % No nonlinear constraints, custom options

    %%


    % Check if fmincon actually improved the result
    if loss_value > min_grid_loss
        warning('fmincon did not improve on the grid search result. Using grid result instead.');
        calibrated_rho = best_grid_rho;
        loss_value = min_grid_loss;
    end


end