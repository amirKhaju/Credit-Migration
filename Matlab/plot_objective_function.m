function plot_objective_function(observed_joint_probs, z_BBB, z_A, mode)
    % This function plots the objective function for rho values between 0
    % and 0.2
    
    objective = @(rho) loss_function(rho, observed_joint_probs, z_BBB, z_A, mode);
    
    % Define the range of rho values to evaluate
    rho_values = linspace(0.001, 0.1, 100);  % 100 points between 0.001 and 0.999
    
    % Calculate loss for each rho value
    loss_values = zeros(size(rho_values));    
    
    for i = 1:length(rho_values)
        loss_values(i) = objective(rho_values(i));
    end
    
    % Find the minimum loss value and corresponding rho
    [min_loss, min_idx] = min(loss_values);
    min_rho = rho_values(min_idx);
    
    % Create a new figure
    figure;
    
    % Plot the objective function
    plot(rho_values, loss_values, 'b-', 'LineWidth', 2);
    hold on;
    
    % Mark the minimum point
    plot(min_rho, min_loss, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    
    xlabel('Correlation Parameter (\rho)', 'FontSize', 12);
    ylabel(['Loss Value (' mode ')'], 'FontSize', 12);
    title(['Objective Function for ' mode ' Loss'], 'FontSize', 14);
    
    grid on;    
    legend('Loss Function', 'Minimum', 'Location', 'best');    
    text(min_rho + 0.05, min_loss, ['Min \rho = ' num2str(min_rho, '%.4f') newline 'Min Loss = ' num2str(min_loss, '%.6e')], ...
        'FontSize', 10, 'VerticalAlignment', 'bottom');
    
end