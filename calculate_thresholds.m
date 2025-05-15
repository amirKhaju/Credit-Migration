function thresholds = calculate_thresholds(transition_matrix)
    cum_probs = fliplr(cumsum(fliplr(transition_matrix), 2));
    cum_probs = min(cum_probs, 0.99999);
    cum_probs = max(cum_probs, 0.00001);
    thresholds = norminv(cum_probs);
end