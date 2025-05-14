clear
close all
clc
warning('off', 'all')
format long

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
cdf_A = cumsum(marginal_A);        % 1×8
cdf_BBB = cumsum(marginal_BBB);    % 8×1
z_A = [-Inf, norminv(cdf_A)];       % 1×9: thresholds for A
z_BBB = [-Inf; norminv(cdf_BBB)];   % 9×1: thresholds for BBB

rng(18);  % Set random seed

mc_simulations =1000000;
obligor_num = 7;

%y = randn(mc_simulations, 1);              % Common factor (Y)
%z = randn(mc_simulations, issuers_num);    % Idiosyncratic shocks (ei)


bivariate_normal_cdf = @(x, y, rho) mvncdf([x, y], [0, 0], [1, rho; rho, 1]);

modes = {'MSE', 'MAE', 'likelihood', 'gradient_descent'};
P_results = cell(1, length(modes));

for i = 1:length(modes)
    mode = modes{i};
    [calibrated_rho, loss_value] = calibrate_rho(empirical_joint_prob, z_BBB, z_A, mode);
    P_results{i} = calculate_theoretical_joint_probs(z_BBB, z_A, calibrated_rho);
    plot_objective_function(empirical_joint_prob, z_BBB, z_A, mode);
end


