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
N_A = 50;       % number of bonds with rating A
N_BBB = 50;     % number of bonds with rating BBB
recovery_rate=0.4;      % setting recovery rate = 40%

today = datetime(2023,2,2);

% Generate payment dates
ndate=5;
datepag = today + calmonths(6*(0:(ndate-1)));
datepag=datenum(datepag);
datepag=adjust_to_business_days(datepag);


% Compute discount factors for the payment dates of the bond
df_expiry = getDiscount(dates, discounts,datepag);


% Transition matrix P (rows = initial rating, columns = destination rating)
% Ratings: AAA, AA, A, BBB, BB, B, CCC, Default

transition_matrix = [
    0.9005, 0.0918, 0.0055, 0.0005, 0.0008, 0.0003, 0.0005, 0.0000;  % from AAA
    0.0057, 0.9006, 0.0861, 0.0058, 0.0006, 0.0007, 0.0002, 0.0002;  % from AA
    0.0003, 0.0196, 0.9165, 0.0575, 0.0037, 0.0015, 0.0002, 0.0007;  % from A
    0.0001, 0.0013, 0.0383, 0.9096, 0.0408, 0.0063, 0.0014, 0.0022;  % from BBB
    0.0002, 0.0004, 0.0017, 0.0576, 0.8450, 0.0785, 0.0076, 0.0089;  % from BB
    0.0000, 0.0003, 0.0012, 0.0025, 0.0621, 0.8368, 0.0505, 0.0465;  % from B
    0.0000, 0.0000, 0.0018, 0.0027, 0.0081, 0.1583, 0.5140, 0.3152   % from CCC
];
transition_matrix = [transition_matrix; [zeros(1,7),1]];

%%
% Parameters
issuers_num_A = 50;  % Number of bonds in the portfolio with rating A
issuers_num_BBB = 50;  % Number of bonds in the portfolio with rating BBB
maturity = 2;  % Maturity in years

rating_AAA=1;
rating_AA=2;
rating_A = 3; 
rating_BBB= 4;
rating_BB= 5;
rating_B= 6;
rating_CCC= 7;
rating_def = 8;
% Compute bond expiry date
expiry = datetime(2025,2,3);

transition_matrix2y=transition_matrix^2;


recovery_rate = 0.4;     % Recovery rate in the event of default
face_value = 1;  % Face value of each bond
rho = calibrated_rho;

mc_simulations = 1e6;  % Number of Monte Carlo simulation scenarios

df_expiry = getDiscount(dates, discounts,datepag);
df_expiry(1)=1;                     

% Extract the discount factor at the bond's final maturity
discexp=df_expiry(end);

frwdisc1y2y=df_expiry(5)/df_expiry(3);      % B(0,1,2)=B(0,2)/B(0,1)
% Compute the 0-0.5-year forward discount factor
frwdis6mesi=df_expiry(2);           % B(0,0,1/2)=B(0,1/2)/B(0,0)

%%
% default prob 1 year (starting from x)
pd_1y_AAA = transition_matrix(rating_AAA,rating_def);
pd_1y_AA = transition_matrix(rating_AA,rating_def);  
pd_1y_A = transition_matrix(rating_A,rating_def);     
pd_1y_BBB = transition_matrix(rating_BBB,rating_def);
pd_1y_BB = transition_matrix(rating_BB,rating_def);  
pd_1y_B = transition_matrix(rating_B,rating_def);  
pd_1y_CCC = transition_matrix(rating_CCC,rating_def);  
pd_1y_def = transition_matrix(rating_def,rating_def);

%% Compute the 1-year discount factor considering default probability and recovery
df_1y_def_A=(1-pd_1y_A)*df_expiry(3)+pd_1y_A*frwdis6mesi*recovery_rate;
df_1y_def_BBB=(1-pd_1y_BBB)*df_expiry(3)+pd_1y_BBB*frwdis6mesi*recovery_rate;

%%
% Compute DF(1y→2y | AAA)
df_1y2ydef_AAA = (1-pd_1y_AAA)*frwdisc1y2y+pd_1y_AAA*df_expiry(4)/df_expiry(3)*recovery_rate;
% Compute DF(1y→2y | AA)
df_1y2ydef_AA = (1-pd_1y_AA)*frwdisc1y2y+pd_1y_AA*df_expiry(4)/df_expiry(3)*recovery_rate;
% Compute DF(1y→2y | A)
df_1y2ydef_A = (1-pd_1y_A)*frwdisc1y2y+pd_1y_A*df_expiry(4)/df_expiry(3)*recovery_rate;
% Compute DF(1y→2y | BBB)
df_1y2ydef_BBB = (1-pd_1y_BBB)*frwdisc1y2y+pd_1y_BBB*df_expiry(4)/df_expiry(3)*recovery_rate;
% Compute DF(1y→2y | BB)
df_1y2ydef_BB = (1-pd_1y_BB)*frwdisc1y2y+pd_1y_BB*df_expiry(4)/df_expiry(3)*recovery_rate;
% Compute DF(1y→2y | B)
df_1y2ydef_B = (1-pd_1y_B)*frwdisc1y2y+pd_1y_B*df_expiry(4)/df_expiry(3)*recovery_rate;
% Compute DF(1y→2y | CCC)
df_1y2ydef_CCC = (1-pd_1y_CCC)*frwdisc1y2y+pd_1y_CCC*df_expiry(4)/df_expiry(3)*recovery_rate;

%%


% 
% Compute the expected discount factor at 2y, taking into account all rating transitions from A
df_2y_def_A = (...
        transition_matrix(rating_A,rating_AAA)*df_expiry(3)*df_1y2ydef_AAA + ...
        transition_matrix(rating_A,rating_AA)*df_expiry(3)*df_1y2ydef_AA +  ...
        transition_matrix(rating_A,rating_A)*df_expiry(3)*df_1y2ydef_A + ...
        transition_matrix(rating_A,rating_BBB)*df_expiry(3)*df_1y2ydef_BBB +  ...
        transition_matrix(rating_A,rating_BB)*df_expiry(3)*df_1y2ydef_BB +  ...
        transition_matrix(rating_A,rating_B)*df_expiry(3)*df_1y2ydef_B +  ...
        transition_matrix(rating_A,rating_CCC)*df_expiry(3)*df_1y2ydef_CCC + ...
        transition_matrix(rating_A,rating_def)*df_expiry(2)*recovery_rate ...
);

% Compute the expected discount factor at 2y, taking into account all
% rating transitions from BBB
df_2y_def_BBB = (...
        transition_matrix(rating_BBB,rating_AAA)*df_expiry(3)*df_1y2ydef_AAA + ...
        transition_matrix(rating_BBB,rating_AA)*df_expiry(3)*df_1y2ydef_AA +  ...
        transition_matrix(rating_BBB,rating_A)*df_expiry(3)*df_1y2ydef_A + ...
        transition_matrix(rating_BBB,rating_BBB)*df_expiry(3)*df_1y2ydef_BBB +  ...
        transition_matrix(rating_BBB,rating_BB)*df_expiry(3)*df_1y2ydef_BB +  ...
        transition_matrix(rating_BBB,rating_B)*df_expiry(3)*df_1y2ydef_B +  ...
        transition_matrix(rating_BBB,rating_CCC)*df_expiry(3)*df_1y2ydef_CCC + ...
        transition_matrix(rating_BBB,rating_def)*df_expiry(2)*recovery_rate ...
);

% 
%%  -- Forward default probabilities between year 1 and year 2 --
% Compute survival probabilities for A
survival_prob_1y_A =1-pd_1y_A;
survival_prob_2y_A =1-transition_matrix2y(rating_A,rating_def);

% Compute P(def 1-2 | alive in 0-1) for A
forwprob_A=(survival_prob_1y_A-survival_prob_2y_A)/survival_prob_1y_A;

% Compute survival probabilities for BBB
survival_prob_1y_BBB =1-pd_1y_BBB;
survival_prob_2y_BBB =1-transition_matrix2y(rating_BBB,rating_def);

% Compute P(def 1-2 | alive in 0-1) for BBB
forwprob_BBB=(survival_prob_1y_BBB-survival_prob_2y_BBB)/survival_prob_1y_BBB;

%%
% --- Forward bond prices and portfolio mark-to-market ---
% FOR A
% Forward price of a 1y bond (from today t=0), assuming issuer is AAA at t=1
fwd_price_A_AAA = df_1y2ydef_AAA*face_value;
% Forward price of a 1y bond, assuming the issuer is AA at t=1
fwd_price_A_AA = df_1y2ydef_AA*face_value;
% Forward price of a 1y bond, assuming the issuer is A at t=1
fwd_price_A_A = df_1y2ydef_A*face_value;
% Forward price of a 1y bond, assuming the issuer is BBB at t=1
fwd_price_A_BBB = df_1y2ydef_BBB*face_value;
% Forward price of a 1y bond, assuming the issuer is BB at t=1
fwd_price_A_BB = df_1y2ydef_BB*face_value;
% Forward price of a 1y bond, assuming the issuer is B at t=1
fwd_price_A_B = df_1y2ydef_B*face_value;
% Forward price of a 1y bond, assuming the issuer is CCC at t=1
fwd_price_A_CCC = df_1y2ydef_CCC*face_value;

% FOR BBB
% Forward price of a 1y bond (from today t=0), assuming issuer is AAA at t=1
fwd_price_BBB_AAA = df_1y2ydef_AAA*face_value;
% Forward price of a 1y bond, assuming the issuer is AA at t=1
fwd_price_BBB_AA = df_1y2ydef_AA*face_value;
% Forward price of a 1y bond, assuming the issuer is A at t=1
fwd_price_BBB_A = df_1y2ydef_A*face_value;
% Forward price of a 1y bond, assuming the issuer is BBB at t=1
fwd_price_BBB_BBB = df_1y2ydef_BBB*face_value;
% Forward price of a 1y bond, assuming the issuer is BB at t=1
fwd_price_BBB_BB = df_1y2ydef_BB*face_value;
% Forward price of a 1y bond, assuming the issuer is B at t=1
fwd_price_BBB_B = df_1y2ydef_B*face_value;
% Forward price of a 1y bond, assuming the issuer is CCC at t=1
fwd_price_BBB_CCC = df_1y2ydef_CCC*face_value;

%%
% Current mark-to-market of a single 2y bond, accounting for credit risk and transitions
bond_mtm_A = df_2y_def_A*face_value;

bond_mtm_BBB = df_2y_def_BBB*face_value;

% Mark-to-market of the entire portfolio (50 issuers A and 50 issuers BBB)
ptf_mtm = bond_mtm_A*issuers_num_A+bond_mtm_BBB*issuers_num_BBB;



%% Compute the VaR
transMatFull = transition_matrix;  % matrice 7x8 completa

issuers_num_A = 50;
issuers_num_B = 50;
faceValue = 1;
seed = 0;

mc_simulations = 1e6;

df_1y2ydef=[df_1y2ydef_AAA, df_1y2ydef_AA, df_1y2ydef_A, df_1y2ydef_BBB, df_1y2ydef_BB, df_1y2ydef_B, df_1y2ydef_CCC];
flag = 1;  % commenta
VaR_ex2a = compute_VaR_for_rho(0.01,transition_matrix,df_1y2ydef,df_expiry,bond_mtm_A,bond_mtm_BBB,frwdis6mesi,face_value,recovery_rate,mc_simulations,issuers_num_A,issuers_num_BBB,seed,flag,0,0)


%% point 2.b
S= 50;  % 50 million annual sales
k=50;

rho_function = @(pd) 0.12*(1-exp(-k*pd))/(1-exp(-k)) + ...
                     0.24*(1-(1-exp(-k*pd))/(1-exp(-k))) - ...
                     0.04*(1-(S-5)/45);

rho_A=rho_function(pd_1y_A);    % compute rho_A using 1y prob of default starting from rating A
rho_BBB=rho_function(pd_1y_BBB);    % compute rho_A using 1y prob of default starting from rating BBB
flag = 2;  % commenta

VaR_ex2b = compute_VaR_for_rho(0.01,transition_matrix,df_1y2ydef,df_expiry,bond_mtm_A,bond_mtm_BBB,frwdis6mesi,face_value,recovery_rate,mc_simulations,issuers_num_A,issuers_num_BBB,seed,flag,rho_A,rho_BBB)


