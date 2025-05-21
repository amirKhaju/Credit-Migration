function [VaR_ex, lossexd, avgdowngrade]  = compute_VaR_for_rho(rho,transition_matrix,df_1y2ydef,df_expiry,bond_mtm_A,bond_mtm_BBB,face_value,recovery_rate,mc_simulations,issuers_num_A,issuers_num_BBB,seed,flag,rho_A,rho_BBB)
%
% Computes the 99% Monte Carlo Value at Risk (VaR) for a credit portfolio of zero-coupon bonds,
% using a single-factor Gaussian copula model, including default and migration risk.
%
% INPUTS:
%   rho              : scalar, common asset correlation (used if flag == 1)
%   transition_matrix: [8x8] rating transition matrix (1-year horizon)
%   df_1y2ydef       : [8x1] discount factors for each rating in year 2 (including default recovery)
%   df_expiry        : [nx1] vector of discount factors for selected payment dates
%   bond_mtm_A       : mark-to-market price of a 2-year bond initially rated A
%   bond_mtm_BBB     : mark-to-market price of a 2-year bond initially rated BBB
%   face_value       : face value of each bond
%   recovery_rate    : recovery rate in case of default
%   mc_simulations   : number of Monte Carlo simulations
%   issuers_num_A    : number of bonds initially rated A
%   issuers_num_BBB  : number of bonds initially rated BBB
%   seed             : random seed for reproducibility
%   flag             : 1 to use scalar rho, 2 to use rho_A and rho_BBB separately
%   rho_A            : correlation for A-rated issuers (used if flag == 2)
%   rho_BBB          : correlation for BBB-rated issuers (used if flag == 2)
%
% OUTPUTS:
%   VaR_ex           : estimated 99.9% Monte Carlo Value at Risk
%   lossexd          : [mc_simulations x 1] vector of simulated total portfolio losses
%%
% Parameters
rating_A = 3;
rating_BBB = 4;

% Set the seed
rng(seed);

% Generate idiosyncratic shocks for each issuer and a common systematic
% factor
epsilon_A = randn(mc_simulations,issuers_num_A);
epsilon_BBB = randn(mc_simulations,issuers_num_BBB);
y = randn(mc_simulations,1);

if flag == 1
    % Single-factor latent variable model
    v_A = sqrt(rho) * y + sqrt(1 - rho) * epsilon_A;
    v_BBB = sqrt(rho) * y + sqrt(1 - rho) * epsilon_BBB;
elseif flag == 2
    v_A = sqrt(rho_A) * y + sqrt(1 - rho_A) * epsilon_A;
    v_BBB = sqrt(rho_BBB) * y + sqrt(1 - rho_BBB) * epsilon_BBB;
end

%% Compute thresholds 

barriers_downgrade_A = compute_barriers(transition_matrix,rating_A);

barriers_downgrade_BBB = compute_barriers(transition_matrix,rating_BBB);

edges_A = [-Inf, barriers_downgrade_A];        % 8 soglie → 9 intervalli
edges_BBB = [-Inf, barriers_downgrade_BBB];    % stessa cosa
  
states_A = 9 - discretize(v_A, edges_A);             % dimensione: [mc_simulations × issuers_num_A]

states_BBB = 9 - discretize(v_BBB, edges_BBB);     % dimensione: [mc_simulations × issuers_num_BBB]

%% Compute number of defaults and downgrades for issuers with initial rating A
% Conta quanti bond vanno in ciascun rating (1 = AAA, 8 = Default), per ogni simulazione
% columns ii matrix counts: 1 = AAA, 2 = AA, 3 = A, 4 = BBB, 5 = BB, 6 = B, 7 = CCC, 8 = Default

counts_A = zeros(mc_simulations, 8);
counts_BBB = zeros(mc_simulations, 8);

for ii = 1:8
    counts_A(:, ii) = sum(states_A == ii, 2);  % Conta quante volte rating i compare per ogni riga
    counts_BBB(:, ii) = sum(states_BBB == ii, 2);  % Conta quante volte rating i compare per ogni riga
end


%%
% vecstay_A = vecdown_A_A;

% Compute average number of downgrades and defaults
% 1 = AAA, 2 = AA, 3 = A, 4 = BBB, 5 = BB, 6 = B, 7 = CCC, 8 = Default
avgdowngrade_A = mean(counts_A(:,1:8));
avgdowngrade_BBB = mean(counts_BBB(:,1:8));

avgdowngrade=[avgdowngrade_A; avgdowngrade_BBB];

%%
% Compute the 0-0.5-year forward discount factor
frwdis6mesi = df_expiry(2);         % B(0;0,1/2)=B(0,1/2)/B(0,0)

price1y_A = ((counts_A(:,1:7) * df_1y2ydef(:) + ...
                counts_A(:,8) * recovery_rate * frwdis6mesi ...
                )/issuers_num_A) * face_value;

price1y_BBB = ((counts_BBB(:,1:7) * df_1y2ydef(:) + ...
                counts_BBB(:,8) * recovery_rate * frwdis6mesi ...
                )/issuers_num_BBB) * face_value;

%% Compute loss per scenario

lossexd_A = bond_mtm_A / df_expiry(3) - price1y_A;
lossexd_BBB = bond_mtm_BBB / df_expiry(3) - price1y_BBB;

% I evaluate losses in t=0 or t=1 ?
lossexd = (lossexd_A * issuers_num_A + lossexd_BBB * issuers_num_BBB);
% lossexd = (lossexd_A * issuers_num_A + lossexd_BBB * issuers_num_BBB)*df_expiry(3);

% % Sort losses in descending order to compute VaR
% lossordexd = sort(lossexd, 'descend');
% % Index of the 0.1% quantile
% idx = ceil(0.001 * mc_simulations);
% % Compute VaR
% VaR_ex = lossordexd(idx);


confidence_level=0.999;

% Compute VaR
VaR_ex = prctile(lossexd, confidence_level * 100);


end