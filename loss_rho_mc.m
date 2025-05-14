function err = loss_rho_mc(rho, z_A, z_BBB, empirical)
% LOSS_RHO_MC Estimate the error between empirical and Monte Carlo-simulated joint migration matrix
%
% This function simulates the joint rating transitions of two firms
% (one initially rated A and one rated BBB) under the single-factor Vasicek model
% with a given asset correlation rho. The simulated joint transition matrix is compared
% to an empirical one, and the mean squared error (MSE) is returned.
%
% INPUTS:
%   rho        - Scalar asset correlation between the two firms (0 < rho < 1)
%   z_A        - Vector of cumulative threshold values (length K+1) for A-rated firm
%   z_BBB      - Vector of cumulative threshold values (length K+1) for BBB-rated firm
%   empirical  - Empirical joint migration matrix of size K x K (normalized to probabilities)
%
% OUTPUT:
%   err        - Scalar value representing the mean squared error between the
%                empirical matrix and the simulated one under current rho   
%%

N = 1e6;

% simulation of Vasicek model
rng(1);
Z = randn(N, 1);
epsA = randn(N, 1);
epsB = randn(N, 1);

% X_A = rho * Z + sqrt(1 - rho^2) * epsA;
% X_BBB = rho * Z + sqrt(1 - rho^2) * epsB;

X_A = sqrt(rho) * Z + sqrt(1 - rho) * epsA;
X_BBB = sqrt(rho) * Z + sqrt(1 - rho) * epsB;

rating_A = discretize(X_A, z_A);
rating_BBB = discretize(X_BBB, z_BBB);

K = length(z_A) - 1;
joint_sim = accumarray([rating_BBB, rating_A], 1, [K K]) / N;

% Errore tra matrice simulata e matrice empirica
err = sum((joint_sim(:) - empirical(:)).^2);  % MSE

end
