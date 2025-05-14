function model_joint = vasicek_model_matrix(z_A, z_BBB, rho)
% Computes the model-implied joint migration probability matrix under the
% single-factor Vasicek model for a given asset correlation rho.
%
% INPUTS:
%   z_A     - Vector of latent variable thresholds for A-rated firms (1×9)
%   z_BBB   - Vector of latent variable thresholds for BBB-rated firms (9×1)
%   rho     - Asset correlation (scalar) between the two firms
%
% OUTPUT:
%   model_joint - 8×8 matrix of joint migration probabilities (from BBB × A)
%
%%

% Initialize the 8×8 joint probability matrix
model_joint = zeros(8,8);

for i = 1:8
    for j = 1:8
        % Compute the probability that:
        % BBB-firm lands in [z_BBB(i), z_BBB(i+1)]
        % A-firm   lands in [z_A(j), z_A(j+1)]
        % using the bivariate standard normal CDF
        model_joint(i,j) = bivar_box_prob(z_A(j), z_A(j+1), z_BBB(i), z_BBB(i+1), rho);
    end
end


end
