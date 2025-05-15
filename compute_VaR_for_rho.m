function VaR_ex = compute_VaR_for_rho(rho,transition_matrix,df_1y2ydef,df_expiry,bond_mtm_A,bond_mtm_BBB,frwdis6mesi,face_value,recovery_rate,mc_simulations,issuers_num_A,issuers_num_BBB,seed,flag,rho_A,rho_BBB)
%
% Computes the 99% Monte Carlo Value at Risk (VaR) for a credit portfolio of zero-coupon bonds,
% using a single-factor Gaussian copula model, including default and migration risk.
%%

% Set rating numbers for readability
rating_AAA=1;
rating_AA=2;
rating_A = 3; 
rating_BBB= 4;
rating_BB= 5;
rating_B= 6;
rating_CCC= 7;
rating_def = 8;

% Set the seed
rng(seed);

total_issuers_num = issuers_num_A + issuers_num_BBB;  % total number of bonds (100)

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

%%
% % other model
% if flag == 1
%     % Single-factor latent variable model
%     v_A = rho * y + sqrt(1 - rho^2) * epsilon_A;
%     v_BBB = rho * y + sqrt(1 - rho^2) * epsilon_BBB;
% elseif flag == 2
%     v_A = rho_A * y + sqrt(1 - rho_A^2) * epsilon_A;
%     v_BBB = rho_BBB * y + sqrt(1 - rho_BBB^2) * epsilon_BBB;
% end


%%
% Compute default threshold using inverse CDF for A
barrier_downgrade_A_def=norminv(transition_matrix(rating_A,rating_def));
barrier_downgrade_A_CCC=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_CCC)));
barrier_downgrade_A_B=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_B)));
barrier_downgrade_A_BB=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_BB)));
barrier_downgrade_A_BBB=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_BBB)));
barrier_downgrade_A_A=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_A)));
barrier_downgrade_A_AA=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_AA)));
barrier_downgrade_A_AAA=norminv(sum(transition_matrix(rating_A,rating_def:-1:rating_AAA)));

% Compute default threshold using inverse CDF for BBB
barrier_downgrade_BBB_def=norminv(transition_matrix(rating_BBB,rating_def));
barrier_downgrade_BBB_CCC=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_CCC)));
barrier_downgrade_BBB_B=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_B)));
barrier_downgrade_BBB_BB=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_BB)));
barrier_downgrade_BBB_BBB=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_BBB)));
barrier_downgrade_BBB_A=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_A)));
barrier_downgrade_BBB_AA=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_AA)));
barrier_downgrade_BBB_AAA=norminv(sum(transition_matrix(rating_BBB,rating_def:-1:rating_AAA)));

%% Compute number of defaults and downgrades for issuers with initial rating A

% count the number of defaults for each simulation
vecdef_A = sum(v_A < barrier_downgrade_A_def, 2);    % sum over each row

% count the number of downgrade to CCC for each simulation
vecdown_A_CCC = sum((v_A < barrier_downgrade_A_CCC) & (v_A >= barrier_downgrade_A_def), 2);
% count the number of downgrade to B for each simulation
vecdown_A_B = sum((v_A < barrier_downgrade_A_B) & (v_A >= barrier_downgrade_A_CCC), 2);
% count the number of downgrade to BB for each simulation
vecdown_A_BB = sum((v_A < barrier_downgrade_A_BB) & (v_A >= barrier_downgrade_A_B), 2);
% count the number of downgrade to BBB for each simulation
vecdown_A_BBB = sum((v_A < barrier_downgrade_A_BBB) & (v_A >= barrier_downgrade_A_BB), 2);
% count the number of issuers that remain in A for each simulation
vecdown_A_A = sum((v_A < barrier_downgrade_A_A) & (v_A >= barrier_downgrade_A_BBB), 2);
% count the number of upgrade to AA for each simulation
vecdown_A_AA = sum((v_A < barrier_downgrade_A_AA) & (v_A >= barrier_downgrade_A_A), 2);
% count the number of upgrade to AAA for each simulation
vecdown_A_AAA = sum((v_A < barrier_downgrade_A_AAA) & (v_A >= barrier_downgrade_A_AA), 2);

vectdown_A=[vecdown_A_CCC, vecdown_A_B, vecdown_A_BB, vecdown_A_BBB, vecdown_A_A, vecdown_A_AA,vecdown_A_AAA];

%% Compute number of defaults and downgrades for issuers with initial rating BBB
% count the number of defaults for each simulation
vecdef_BBB = sum(v_BBB < barrier_downgrade_BBB_def, 2);    % sum over each row

% count the number of downgrade to CCC for each simulation
vecdown_BBB_CCC = sum((v_BBB < barrier_downgrade_BBB_CCC) & (v_BBB >= barrier_downgrade_BBB_def), 2);
% count the number of downgrade to B for each simulation
vecdown_BBB_B = sum((v_BBB < barrier_downgrade_BBB_B) & (v_BBB >= barrier_downgrade_BBB_CCC), 2);
% count the number of downgrade to BB for each simulation
vecdown_BBB_BB = sum((v_BBB < barrier_downgrade_BBB_BB) & (v_BBB >= barrier_downgrade_BBB_B), 2);
% count the number of issuers that remain in A for each simulation
vecdown_BBB_BBB = sum((v_BBB < barrier_downgrade_BBB_BBB) & (v_BBB >= barrier_downgrade_BBB_BB), 2);
% count the number of upgrade to A for each simulation
vecdown_BBB_A = sum((v_BBB < barrier_downgrade_BBB_A) & (v_BBB >= barrier_downgrade_BBB_BBB), 2);
% count the number of upgrade to AA for each simulation
vecdown_BBB_AA = sum((v_BBB < barrier_downgrade_BBB_AA) & (v_BBB >= barrier_downgrade_BBB_A), 2);
% count the number of upgrade to AAA for each simulation
vecdown_BBB_AAA = sum((v_BBB < barrier_downgrade_BBB_AAA) & (v_BBB >= barrier_downgrade_BBB_AA), 2);

vectdown_BBB=[vecdown_BBB_CCC, vecdown_BBB_B, vecdown_BBB_BB, vecdown_BBB_BBB, vecdown_BBB_A, vecdown_BBB_AA,vecdown_BBB_AAA];

%%
vecstay_A = vecdown_A_A;

% Compute average number of downgrades
avgdowngrade_CCC = mean(vecdown_A_CCC);
avgdowngrade_B = mean(vecdown_A_B);
avgdowngrade_BB = mean(vecdown_A_BB);
avgdowngrade_BBB = mean(vecdown_A_BBB);
avgdowngrade_A = mean(vecstay_A);
avgdowngrade_AA = mean(vecdown_A_AA);
avgdowngrade_AAA = mean(vecdown_A_AAA);

vecstay_BBB = vecdown_BBB_BBB;

% Compute average number of downgrades
avgdowngrade_CCC = mean(vecdown_A_CCC);
avgdowngrade_B = mean(vecdown_A_B);
avgdowngrade_BB = mean(vecdown_A_BB);
avgdowngrade_BBB = mean(vecdown_A_BBB);
avgdowngrade_A = mean(vecstay_A);
avgdowngrade_AA = mean(vecdown_A_AA);
avgdowngrade_AAA = mean(vecdown_A_AAA);

%%
% Compute average number of defaults
avgdefault_A = mean(vecdef_A);
avgdefault_BBB = mean(vecdef_BBB);

price1y_A = ((vecdef_A * recovery_rate * frwdis6mesi + ...
                vecdown_A_CCC * df_1y2ydef(7) + ...
                vecdown_A_B * df_1y2ydef(6) + ...
                vecdown_A_BB * df_1y2ydef(5) + ...
                vecdown_A_BBB * df_1y2ydef(4) + ...
                vecdown_A_A * df_1y2ydef(3) + ...
                vecdown_A_AA * df_1y2ydef(2) + ...
                vecdown_A_AAA * df_1y2ydef(1) ...
                ) / issuers_num_A) * face_value;

price1y_BBB = ((vecdef_BBB * recovery_rate * frwdis6mesi + ...
                vecdown_BBB_CCC * df_1y2ydef(7) + ...
                vecdown_BBB_B * df_1y2ydef(6) + ...
                vecdown_BBB_BB * df_1y2ydef(5) + ...
                vecdown_BBB_BBB * df_1y2ydef(4) + ...
                vecdown_BBB_A * df_1y2ydef(3) + ...
                vecdown_BBB_AA * df_1y2ydef(2) + ...
                vecdown_BBB_AAA * df_1y2ydef(1) ...
                ) / issuers_num_BBB) * face_value;

%%
% bond_mtm=bond_mtm_A+bond_mtm_BBB;
% Compute loss per scenario
lossexd_A = bond_mtm_A / df_expiry(3) - price1y_A;
lossexd_BBB = bond_mtm_BBB / df_expiry(3) - price1y_BBB;

% lossexd=[lossexd_A; lossexd_BBB];   % size is (2000000x1)

lossexd = lossexd_A * issuers_num_A + lossexd_BBB * issuers_num_BBB;

% Sort losses in descending order to compute VaR
lossordexd = sort(lossexd, 'descend');


% Index of the 0.1% quantile
idx = ceil(0.001 * mc_simulations);

% Compute VaR
VaR_ex = lossordexd(idx);


end