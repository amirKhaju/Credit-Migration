function joint_probs = calculate_theoretical_joint_probs(z_BBB, z_A, rho)
    % Create matrices for all combinations of indices
    [I, J] = ndgrid(1:8, 1:8);
    
    % Get the z values for all combinations
    z_BBB_i = z_BBB(I);
    z_BBB_i_plus_1 = z_BBB(I+1);
    z_A_j = z_A(J);
    z_A_j_plus_1 = z_A(J+1);
    
    % Create correlation matrix for mvncdf
    corr_matrix = [1 rho; rho 1];
    mu = [0, 0];
    
    % For MATLAB's mvncdf

    P11 = reshape(mvncdf([z_BBB_i_plus_1(:), z_A_j_plus_1(:)], mu, corr_matrix), size(I));
    P10 = reshape(mvncdf([z_BBB_i_plus_1(:), z_A_j(:)], mu, corr_matrix), size(I));
    P01 = reshape(mvncdf([z_BBB_i(:), z_A_j_plus_1(:)], mu, corr_matrix), size(I));
    P00 = reshape(mvncdf([z_BBB_i(:), z_A_j(:)], mu, corr_matrix), size(I));
    
    % Calculate the final probabilities matrix
    joint_probs = P11 - P10 - P01 + P00;
end