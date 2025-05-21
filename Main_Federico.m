% Final Project Group 4.A 
% 
% AY2024-2025 
% 
% Federico Favali
% Amirreza Khajouei
%

% Clear workspace and close all figures
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

% load joint credit movements matrix
load joint_counts.mat

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

% rng(18);  % Set random seed

% mc_simulations =1000000;
% obligor_num = 7;

%y = randn(mc_simulations, 1);              % Common factor (Y)
%z = randn(mc_simulations, issuers_num);    % Idiosyncratic shocks (ei)


bivariate_normal_cdf = @(x, y, rho) mvncdf([x, y], [0, 0], [1, rho; rho, 1]);

modes = {'MSE', 'MAE', 'likelihood', 'gradient_descent'};
P_results = cell(1, length(modes));

%modes = {'MSE', 'MAE', 'likelihood', 'gradient_descent', 'weighted_MAE'};
modes = {'weighted_MAE'};
% for i = 1:length(modes)
%     mode = modes{i};
%     [calibrated_rho, loss_value] = calibrate_rho(empirical_joint_prob, z_BBB, z_A, mode);
%     plot_objective_function(empirical_joint_prob, z_BBB, z_A, mode);
% end


%% point 2
close all
clc
load rateSet.mat
load datesSet.mat
load cSelect.mat

%% Settings
formatData='dd/mm/yyyy'; %Pay attention to your computer settings 

%% Read market data
% Reads the market data from an Excel file.
% note: This fuction works on Windows OS. Pay attention on other OS.

%[datesSet, ratesSet] = readExcelData('MktData_CurveBootstrap', formatData);

%% Bootstrap
% computes discount factors from market rates with bootstrap method
% dates includes SettlementDate as first date

[dates, discounts]=bootstrap(datesSet, ratesSet);   % Compute discount factors with bootstrap


%%

issuers_num_A = 50;  % Number of bonds in the portfolio with rating A
issuers_num_BBB = 50;  % Number of bonds in the portfolio with rating BBB
recovery_rate = 0.4;     % Recovery rate in the event of default
face_value = 1;  % Face value of each bond

today = datetime(2023,2,2);

% Generate payment dates
ndate=5;
datepag = today + calmonths(6*(0:(ndate-1)));
datepag=datenum(datepag);
datepag=adjust_to_business_days(datepag);


% Compute discount factors for the payment dates of the bond
df_expiry = getDiscount(dates, discounts,datepag);
df_expiry(1)=1;

% Transition matrix P (rows = initial rating, columns = destination rating)
% Ratings: AAA, AA, A, BBB, BB, B, CCC, Default
load transition_matrix.mat


%%
% Parameters
rating_AAA = 1;
rating_AA = 2;
rating_A = 3; 
rating_BBB = 4;
rating_BB = 5;
rating_B = 6;
rating_CCC = 7;
rating_def = 8;

%%

% Compute the 2 years transition matrix
transition_matrix2y=transition_matrix^2;                   

% Compute the 1-2-year forward discount factor
frwdisc1y2y=df_expiry(5)/df_expiry(3);      % B(0;1,2)=B(0,2)/B(0,1)

% Compute the 0-0.5-year forward discount factor
frwdis6mesi=df_expiry(2);           % B(0;0,1/2)=B(0,1/2)/B(0,0)

%% default prob 1 year (starting from x)
 
pd_1y = transition_matrix(rating_AAA:rating_CCC,rating_def);


%% Compute the 1-year discount factor considering default probability and recovery

df_1y_def_A=(1-pd_1y(rating_A))*df_expiry(3)+pd_1y(rating_A)*frwdis6mesi*recovery_rate;
df_1y_def_BBB=(1-pd_1y(rating_BBB))*df_expiry(3)+pd_1y(rating_BBB)*frwdis6mesi*recovery_rate;

%% Compute DF(1y→2y | x)
% where x can be AAA, AA, A, BBB, BB, B, CCC

frwdisc1y_1y6m = df_expiry(4)/df_expiry(3);     % B(0;1,3/2) = B(0,3/2)/B(0,1)

df_1y2ydef = (1-pd_1y)*frwdisc1y2y+pd_1y*frwdisc1y_1y6m*recovery_rate;


%% Compute the expected discount factor at 2y, taking into account all rating transitions
% from A
df_survival_A = transition_matrix(rating_A,rating_AAA:rating_CCC)*df_expiry(3)*df_1y2ydef;
df_default_A = transition_matrix(rating_A,rating_def)*df_expiry(2)*recovery_rate;
df_2y_def_A = df_survival_A + df_default_A;

% from BBB
df_survival_BBB = transition_matrix(rating_BBB,rating_AAA:rating_CCC)*df_expiry(3)*df_1y2ydef;
df_default_BBB = transition_matrix(rating_BBB,rating_def)*df_expiry(2)*recovery_rate;
df_2y_def_BBB = df_survival_BBB + df_default_BBB;


%%  -- Forward default probabilities between year 1 and year 2 --
% Compute survival probabilities for A
survival_prob_1y_A =1-pd_1y(rating_A);
survival_prob_2y_A =1-transition_matrix2y(rating_A,rating_def);

% Compute P(def 1-2 | alive in 0-1) for A
forwprob_A=(survival_prob_1y_A-survival_prob_2y_A)/survival_prob_1y_A;

% Compute survival probabilities for BBB
survival_prob_1y_BBB =1-pd_1y(rating_BBB);
survival_prob_2y_BBB =1-transition_matrix2y(rating_BBB,rating_def);

% Compute P(def 1-2 | alive in 0-1) for BBB
forwprob_BBB=(survival_prob_1y_BBB-survival_prob_2y_BBB)/survival_prob_1y_BBB;

%% Forward bond prices and portfolio mark-to-market

fwd_price = df_1y2ydef * face_value;

%%
% Current mark-to-market of a single 2y bond, accounting for credit risk and transitions
bond_mtm_A = df_2y_def_A*face_value;
bond_mtm_BBB = df_2y_def_BBB*face_value;

% Mark-to-market of the entire portfolio (50 issuers A and 50 issuers BBB)
ptf_mtm = bond_mtm_A*issuers_num_A+bond_mtm_BBB*issuers_num_BBB;



%% Compute the VaR
% rho = calibrated_rho;

seed = 0;   % Set the seed for reproducibility 

mc_simulations = 1e6;   % Set the number of Monte Carlo simulations

flag = 1;  % flag = 1 to use a single value of rho
[VaR_ex2a, losses_ex2a, avgdowngrade_ex2a] = compute_VaR_for_rho(0.01,transition_matrix,df_1y2ydef,df_expiry,bond_mtm_A,bond_mtm_BBB,face_value,recovery_rate,mc_simulations,issuers_num_A,issuers_num_BBB,seed,flag,0,0);
fprintf('VaR with constant rho (point 2.a):   %.7f\n', VaR_ex2a);

%% point 2.b
S= 50;  % 50 million annual sales
k=50;

rho_function = @(pd) 0.12*(1-exp(-k*pd))/(1-exp(-k)) + ...
                     0.24*(1-(1-exp(-k*pd))/(1-exp(-k))) - ...
                     0.04*(1-(S-5)/45);

rho_A=rho_function(pd_1y(rating_A));    % compute rho_A using 1y prob of default starting from rating A
rho_BBB=rho_function(pd_1y(rating_BBB));    % compute rho_A using 1y prob of default starting from rating BBB

flag = 2;  % flag = 2 to use 2 values of rho: rho_A and rho_BBB
[VaR_ex2b, losses_ex2b, avgdowngrade_ex2b] = compute_VaR_for_rho(0.01,transition_matrix,df_1y2ydef,df_expiry,bond_mtm_A,bond_mtm_BBB,face_value,recovery_rate,mc_simulations,issuers_num_A,issuers_num_BBB,seed,flag,rho_A,rho_BBB);
fprintf('VaR with Basel rho function (point 2.b):   %.7f\n', VaR_ex2b);

%%


