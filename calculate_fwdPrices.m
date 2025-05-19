function forwardPrices = calculate_fwdPrices(defaults, recovery_rate, discounts)
    Default_case= recovery_rate * discounts(1);
    fwdPrices   =  (discounts(4)/discounts(2) .* (1 - defaults) + ...
               discounts(3)/discounts(2) .* defaults .*recovery_rate)';
    forwardPrices = [fwdPrices, Default_case];
end