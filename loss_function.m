function loss = loss_function(rho, observed_joint_probs, z_BBB, z_A, mode)
    theoretical_probs = calculate_theoretical_joint_probs(z_BBB, z_A, rho);

    switch mode
        case 'MSE'
            loss = sum(sum((theoretical_probs - observed_joint_probs).^2));
        
        case 'MAE'
            loss = sum(sum(abs(theoretical_probs - observed_joint_probs)));
        
        case 'likelihood'
            epsilon = 1e-10;
            mask = observed_joint_probs > 0;
            log_likelihood = sum(observed_joint_probs(mask) .* log(theoretical_probs(mask) + epsilon));
            loss = -log_likelihood;
        
        case 'gradient_descent'
            loss = sum(sum((theoretical_probs - observed_joint_probs).^2)); % Use MSE for gradient descent
        
        case 'weighted_MSE'
            indices = 1:8;
            weights = indices' * indices;
            weights = weights/sum(sum(weights));
            weights = observed_joint_probs;
            % Calculate weighted MSE
            squared_errors = (theoretical_probs - observed_joint_probs).^2;
            weighted_errors = weights .* squared_errors;
            loss = sum(sum(weighted_errors));
        case 'weighted_MAE'
            % This gives more importance to more extreme scenarios (higher columns)
            [rows, cols] = size(observed_joint_probs);
            %indices = 1:cols;
            %col_weights = exp(indices/8);
            %weights = repmat(col_weights, rows, 1);

            indices = 1:(cols);
            weights = ((indices)' * indices);
            weights = weights/sum(sum(weights));
            errors = abs(theoretical_probs - observed_joint_probs);
            weighted_errors = weights .* errors;
            loss = sum(sum(weighted_errors));
        otherwise
            error('Unsupported loss function');
    end
end