function initial_value = compute_initial_portfolio_value(transition_matrix, recovery_rate, discounts)
    transition_matrix2y = [transition_matrix; zeros(1, 7), 1]^2;
    pd_A = transition_matrix2y(3, end);     % A-rated (row 3)
    pd_BBB = transition_matrix2y(4, end);   % BBB-rated (row 4)

    % Compute bond prices using expected loss
    price_A = ((1 - pd_A)* discounts(4)+(pd_A * recovery_rate)* discounts(2));
    price_BBB = ((1 - pd_BBB)* discounts(4)+(pd_BBB * recovery_rate)* discounts(2));

    % Total portfolio value: 50 A-rated + 50 BBB-rated bonds
    initial_value = 50 * price_A + 50 * price_BBB;
end

