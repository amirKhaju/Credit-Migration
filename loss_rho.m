function err = loss_rho(rho, z_A, z_BBB, empirical,flag)
% Computes the loss function to calibrate the asset correlation rho in the Vasicek single-factor model.
%
% INPUTS:
%   rho        - Asset correlation (scalar) to be calibrated
%   z_A        - Vector of latent variable thresholds for A-rated firms (1×9)
%   z_BBB      - Vector of latent variable thresholds for BBB-rated firms (9×1)
%   empirical  - Empirical joint migration probability matrix (8×8)
%
% OUTPUT:
%   err        - Scalar loss value (sum of squared differences between model and empirical probabilities)
%%

% Compute the model-implied joint migration probability matrix
model = vasicek_model_matrix(z_A, z_BBB, rho);

% Compute error between model and empirical matrices
if flag == 1      % MSE
    err = sum((model(:) - empirical(:)).^2);

elseif flag == 2   % MAE 
    err = sum(abs(model(:) - empirical(:)));

elseif flag == 3    % likelihood
    % Likelihood-based loss (negative log-likelihood)
    joint_counts = empirical * sum(empirical(:));  % convert probabilities back to counts
    eps_val = 1e-10;
    model = max(model, eps_val);  % avoid log(0)
    err = -sum(joint_counts(:) .* log(model(:)));  % negative log-likelihood

elseif flag == 4    % KL Divergence (serve?)
    eps_val = 1e-10;
    emp = empirical + eps_val;
    mod = model + eps_val;
    err = sum(emp(:) .* log(emp(:) ./ mod(:)));
else
    error("Error: Incorrect flag!");
end


end
