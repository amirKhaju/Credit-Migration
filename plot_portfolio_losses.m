function plot_portfolio_losses(portfolio_losses, var, confidence_level)
    % Calculate mean loss
    mean_loss = mean(portfolio_losses);
    
    % Create figure
    figure;
    
    % Plot histogram with KDE overlay
    histogram(portfolio_losses, 100, 'Normalization', 'probability', 'FaceColor', [0.3 0.5 0.7], 'FaceAlpha', 0.6);
    hold on;
    
    % Add VaR line
    var_line = line([var var], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-');
    
    % Add mean loss line
    mean_line = line([mean_loss mean_loss], ylim, 'Color', 'g', 'LineWidth', 2, 'LineStyle', '--');
    
    % Add labels and title
    xlabel('Portfolio Loss');
    ylabel('Probability');
    title(['Portfolio Loss Distribution with VaR at ' num2str(confidence_level*100) '% Confidence Level']);
    
    % Add legend
    legend([var_line, mean_line], {['VaR at ' num2str(confidence_level*100) '% = ' num2str(var, '%.4f')], ...
           ['Mean Loss = ' num2str(mean_loss, '%.4f')]}, 'Location', 'northeast');
    
    % Add grid
    grid on;
    
    % Enhance appearance
    set(gca, 'FontSize', 12);
    box on;
    
    hold off;
end