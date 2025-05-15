function [var, portfolio_losses] = calculate_var(transition_matrix, discounts, rho, M, confidence_level, recovery_rate)

    forward_1y_2y = discounts(2) / discounts(1);    
    num_ratings = size(transition_matrix, 1) - 1;
    rating_spreads = zeros(num_ratings, 1);
    
    pd = transition_matrix(1:end, end);
    rating_spreads = -log(1 - (1 - recovery_rate) .* pd);
    
    % Step 3: Calculate initial bond prices
    initial_price_A = discounts(2) * exp(-rating_spreads(3) * 2)
    initial_price_BBB = discounts(2) * exp(-rating_spreads(4) * 2)

    initial_portfolio_value = 50 * initial_price_A + 50 * initial_price_BBB;

    cum_probs = fliplr(cumsum(fliplr(transition_matrix), 2));
    thresholds = norminv(cum_probs); % used in rating migration
    
    % Run Monte Carlo simulation
    portfolio_losses = zeros(M, 1);
    Y = randn(M, 1);           % Common risk factor
    e_A = randn(M, 50);        % Idiosyncratic risks for A-rated
    e_BBB = randn(M, 50);      % Idiosyncratic risks for BBB-rated

    % Step 2: Calculate all v values (M x 50)
    v_A = sqrt(rho) * Y + sqrt(1 - rho) * e_A;        % M x 50
    v_BBB = sqrt(rho) * Y + sqrt(1 - rho) * e_BBB;  
    

    compute_bond_values = @(v_mat, threshold_row) arrayfun(@(m) ...
        sum(calculate_bond_values(v_mat(m, :)', threshold_row, rating_spreads, ...
                                  forward_1y_2y, recovery_rate, num_ratings)),(1:M)');


    portfolio_value_A = compute_bond_values(v_A, thresholds(3, 1:num_ratings));
    portfolio_value_B = compute_bond_values(v_BBB, thresholds(4, 1:num_ratings));

    portfolio_value_1y = portfolio_value_A + portfolio_value_B;
    portfolio_losses = initial_portfolio_value - portfolio_value_1y;
    var = prctile(portfolio_losses, confidence_level * 100);
end