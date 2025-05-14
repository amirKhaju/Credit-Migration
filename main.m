clear
close all
clc

%%

% Joint migration matrix from CreditMetrics Table 8.2
% Rows: Final rating of BBB-rated firm
% Columns: Final rating of A-rated firm
% Ratings: AAA, AA, A, BBB, BB, B, CCC, Default

joint_counts = [
     0,     0,     0,     0,     0,     0,    0,    0;
     0,    15,  1105,    54,     4,     0,    0,    0;
     0,   978, 44523,  2812,   414,   224,    0,    0;
     0, 12436,621477, 40584,  5075,  2507,    0,    0;
     0,   839, 41760,  2921,   321,   193,    0,    0;
     0,   175,  7081,   532,    76,    48,    0,    0;
     0,    55,  2230,   127,    18,    15,    0,    0;
     0,    29,   981,    67,     7,     0,    0,    0
];

% total number of cases
N=sum(sum(joint_counts));

% Historically tabulated joint credit quality co-movement
empirical_joint_prob=joint_counts/N;


marginal_A = sum(empirical_joint_prob, 1);    % sum over rows
marginal_BBB = sum(empirical_joint_prob, 2);  % sum over columns


% Cumulative distributions
cdf_A = cumsum(marginal_A);        % (1×8) cumulative probability that an A-rated firm transitions to rating j or better
cdf_BBB = cumsum(marginal_BBB);    % (8×1) cumulative probability that a BBB-rated firm transitions to rating i or better

%
% Add or remove a small value eps_val to avoid 0 and 1 in cdf vectors
eps_val = 1e-14;
cdf_A = min(max(cdf_A, eps_val), 1 - eps_val);
cdf_BBB = min(max(cdf_BBB, eps_val), 1 - eps_val);

% Add -Inf as lower bound and compute z-scores
z_A = [-Inf, norminv(cdf_A)];
z_BBB = [-Inf; norminv(cdf_BBB)];

%
% Add -Inf as lower bound and compute z-scores
% z_A = [-Inf, norminv(cdf_A)];       % (1×9): thresholds for A
% z_BBB = [-Inf; norminv(cdf_BBB)];   % (9×1): thresholds for BBB


lb=0.001;  % lower bound
ub=0.99;   % upper bound

rho_optimal_fminbnd = fminbnd(@(rho) loss_rho(rho, z_A, z_BBB, empirical_joint_prob,1), lb, ub);

rho0 = 0.05; % Set the starting point
rho_optimal_fmincon = fmincon(@(rho) loss_rho(rho, z_A, z_BBB, empirical_joint_prob,1), ...
                      rho0, [], [], [], [], lb, ub);

rho_likelihood_fminbnd = fminbnd(@(rho) loss_rho(rho, z_A, z_BBB, empirical_joint_prob, 4), lb, ub);



model_matrix = vasicek_model_matrix(z_A, z_BBB, rho_optimal_fminbnd);


disp('Optimal rho (with fminbnd):');
disp(rho_optimal_fminbnd);
fprintf("Optimal rho Likelihood (with fminbnd): %.4f\n", rho_likelihood_fminbnd);

disp('Optimal rho (with fmincon):');
disp(rho_optimal_fmincon);


% disp('Empirical matrix:');
% disp(empirical_joint_prob);

% disp('Model matrix at optimal rho:');
% disp(model_matrix);


%% Monte Carlo
% WRONG!
lb = 0.001; ub = 0.99;
rho_opt_mc = fminbnd(@(rho) loss_rho_mc(rho, z_A, z_BBB, empirical_joint_prob), lb, ub);

fprintf("Monte Carlo calibrated rho: %.5f\n", rho_opt_mc);
