function print_all_rho_results(P_results)

    for i=1:length(P_results)
        fprintf('Mode: %-12s | Calibrated ρ = %.6f | Loss = %.6e | CI = [%.6f, %.6f] | std_err = %.6f\n', P_results{i}.mode, P_results{i}.rho, P_results{i}.loss, P_results{i}.CI(1), P_results{i}.CI(2), P_results{i}.stderr);
    end

end