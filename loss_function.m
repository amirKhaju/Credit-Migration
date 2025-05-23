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
        
        case 'KL'
            epsilon = 1e-10;
            % Avoid division by zero and log(0)
            P = observed_joint_probs(:);
            Q = max(theoretical_probs(:), epsilon); 
            
            mask = P > 0;  % Only compute where observed > 0
            loss = sum(P(mask) .* log(P(mask) ./ Q(mask)));
        case 'JSD'
            % Jensen–Shannon divergence
            epsilon = 1e-10;
            P = observed_joint_probs(:);
            Q = theoretical_probs(:);
            % ensure no zeros
            Q = max(Q, epsilon);
            M = 0.5 * (P + Q);
            mask = M > 0;  % only where M>0
            % D_JS = ½ KL(P||M) + ½ KL(Q||M)
            D1 = sum(P(mask) .* log((P(mask) + epsilon) ./ (M(mask) + epsilon)));
            D2 = sum(Q(mask) .* log((Q(mask) + epsilon) ./ (M(mask) + epsilon)));
            loss = 0.5 * (D1 + D2);

        case 'weighted MSE'
            [i_idx, j_idx] = ndgrid(1:8, 1:8);
            weights = i_idx + j_idx;
            % Calculate weighted MSE
            squared_errors = (theoretical_probs - observed_joint_probs).^2;
            weighted_errors = weights .* squared_errors;
            loss = sum(sum(weighted_errors));

        case 'weighted MAE'
            [i_idx, j_idx] = ndgrid(1:8, 1:8);
            weights = i_idx + j_idx;
            errors = abs(theoretical_probs - observed_joint_probs);
            weighted_errors = weights .* errors;
            loss = sum(sum(weighted_errors));
        otherwise
            error('Unsupported loss function');
    end
end