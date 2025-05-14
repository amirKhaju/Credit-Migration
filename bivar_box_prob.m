function p = bivar_box_prob(z1_low, z1_high, z2_low, z2_high, rho)
    % bivar_box_prob computes the joint probability that a bivariate standard
    % normal vector with correlation rho falls within a rectangular region.
    %
    % INPUTS:
    %   z1_low   - Lower bound of the interval for X1
    %   z1_high  - Upper bound of the interval for X1
    %   z2_low   - Lower bound of the interval for X2
    %   z2_high  - Upper bound of the interval for X2
    %   rho      - Correlation between X1 and X2 (rho ∈ [-1, 1])
    %
    % OUTPUT:
    %   p        - Probability that (X1, X2) ∈ [z1_low, z1_high] × [z2_low, z2_high]
    %
    %%
    % Mean vector of the bivariate normal distribution
    mu = [0 0];

    % Covariance matrix for standard normal variables with correlation rho
    Sigma = [1 rho; rho 1];

    % Compute cumulative probabilities for the four corners of the rectangle
    p1 = mvncdf([z1_low, z2_low], mu, Sigma);   % P(X1 ≤ z1_low, X2 ≤ z2_low)
    p2 = mvncdf([z1_high, z2_low], mu, Sigma);  % P(X1 ≤ z1_high, X2 ≤ z2_low)
    p3 = mvncdf([z1_low, z2_high], mu, Sigma);  % P(X1 ≤ z1_low, X2 ≤ z2_high)
    p4 = mvncdf([z1_high, z2_high], mu, Sigma); % P(X1 ≤ z1_high, X2 ≤ z2_high)

    % Apply the inclusion-exclusion principle to get the probability over the rectangle
    % Final result: P(z1_low < X1 ≤ z1_high, z2_low < X2 ≤ z2_high)
    p = p4 - p2 - p3 + p1;

end
