function barriers = compute_barriers(transition_matrix,rating_row)
% Computes the asset return thresholds (barriers)
% corresponding to rating transitions for a given initial rating, based on
% a one-year transition probability matrix.
%
% INPUT:
%   transition_matrix : [8x8] rating transition matrix, where rows represent
%                       initial ratings and columns final ratings
%   rating_row        : scalar index indicating the initial rating (row in the matrix)
%
% OUTPUT:
%   barriers : [1x8] vector of default/migration thresholds (from Default to AAA)
%
%%

% Costruisci barriere cumulative (flip: ordine da AAA a Default → Default a AAA)
cdf = cumsum(flip(transition_matrix(rating_row, :)));

% Calcola le soglie come inverse della normale standard
barriers = norminv(cdf);


end