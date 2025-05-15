function bond_values = calculate_bond_values(v, thresholds, discounts, recovery_rate, defaults)
        
    face_value = 1;
    new_ratings = zeros(size(v));
    num_ratings = 7;
    for i = 1:length(v)
        new_ratings(i) = num_ratings + 1; % Default assumption
        for j = 1:num_ratings
            if v(i) > thresholds(j)
                new_ratings(i) = j;
                break;
            end
        end
    end
    
    % Initialize bond values array
    bond_values = zeros(size(v));
    for i = 1:length(v)
        if new_ratings(i) == num_ratings + 1
            bond_values(i) = face_value * recovery_rate;
        else
            bond_values(i) =  discounts(4)/discounts(2) * (1 - defaults(new_ratings(i))) + ...
               discounts(3)/discounts(2) *defaults(new_ratings(i))*recovery_rate;
        end
    end

end