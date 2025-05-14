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
        otherwise
            error('Unsupported loss function');
    end
end