function corr = basel_correlation(rating_index, transition_matrix, annual_sales)
% BASEL_CORRELATION Calculate the correlation based on Basel II formula
%
% Parameters:
% -----------
% rating_index: int
%   The index of the rating in the transition matrix
%
% transition_matrix: matrix
%   The rating transition matrix
%
% annual_sales: double
%   Annual sales in millions of euros (used for size adjustment)
%
% Returns:
% --------
% corr: double
%   The correlation value according to Basel II formula with size adjustment

    % Get default probability for this rating
    pd = transition_matrix(rating_index, end);
    
    % Basel II correlation formula
    corr = 0.12 * ((1 - exp(-50 * pd)) / (1 - exp(-50))) + ...
           0.24 * (1 - (1 - exp(-50 * pd)) / (1 - exp(-50)));
    
    % Size adjustment for annual sales
    % (5 ≤ S ≤ 50, in millions of euros)
    S = min(max(5, annual_sales), 50);
    size_adj = (1 - 0.04 * (S - 5) / 45) / (1 - 0.04);
    
    corr = corr * size_adj;
end