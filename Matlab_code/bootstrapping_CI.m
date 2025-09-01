function [confidence_interval, method_mean, method_se] = bootstrapping_CI(observed_joint_probs, z_BBB, z_A, mode, calibrated_rho)

rho_min = 0;
rho_max = 1;

confidence_level = 0.95;
n_bootstrap = 700;
bootstrap_rhos = zeros(n_bootstrap, 1);
total_size = 789683;

for i = 1:n_bootstrap
    bootstrap_counts = mnrnd(total_size, observed_joint_probs(:));
    bootstrap_probs = reshape(bootstrap_counts, size(observed_joint_probs)) / total_size;
    objective = @(rho) loss_function(rho, bootstrap_probs, z_BBB, z_A, mode);
    rho0 = calibrated_rho; %starting point
    options = optimoptions('fmincon', 'Display', 'off');
    bootstrap_rhos(i) = fmincon(objective, rho0, [], [], [], [], rho_min, rho_max, [],options);
end

% Calculate confidence interval

alpha = 1 - confidence_level;
lower_percentile = 100 * (alpha/2);
upper_percentile = 100 * (1 - alpha/2);
ci_lower = prctile(bootstrap_rhos, lower_percentile);
ci_upper = prctile(bootstrap_rhos, upper_percentile);

confidence_interval = [ci_lower, ci_upper];

% Display results

method_mean = mean(bootstrap_rhos);
method_se = std(bootstrap_rhos);



fprintf('%.1f%% Confidence Interval: [%.6f, %.6f]\n', ...
    confidence_level*100, ci_lower, ci_upper);
fprintf('Bootstrap std error: %.6f\n', std(bootstrap_rhos));


% hypothesis testing 

null_value = 0;
if mean(bootstrap_rhos) > null_value
    p_value = 2 * sum(bootstrap_rhos <= null_value) / n_bootstrap;
else
    p_value = 2 * sum(bootstrap_rhos >= null_value) / n_bootstrap;
end

fprintf('Bootstrap p-value: %.6f\n', p_value);
if p_value < (1 - confidence_level)
    fprintf('Reject H0 at %.1f%% confidence level (p-value method)\n', confidence_level*100);
else
    fprintf('Fail to reject H0 at %.1f%% confidence level (p-value method)\n', confidence_level*100);
end


% Additional statistics

% fprintf('\nBootstrap Statistics:\n');
% fprintf('Mean: %.6f\n', mean(bootstrap_rhos));
% fprintf('Median: %.6f\n', median(bootstrap_rhos));
% fprintf('Std Dev: %.6f\n', std(bootstrap_rhos));
% fprintf('Min: %.6f, Max: %.6f\n', min(bootstrap_rhos), max(bootstrap_rhos));



end