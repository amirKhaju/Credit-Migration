function compareMethods_with_refrence(method_name, ref_method_name, ...
    method_mean, ref_method_mean, method_se, ref_method_se, bootstrap_num)

% Calculate the difference and its standard error
diff_mean = method_mean - ref_method_mean;
diff_se = sqrt(method_se^2 + ref_method_se^2);

% Calculate t-statistic and p-value
t_stat = diff_mean / diff_se;
p_value = 2 * (1 - tcdf(abs(t_stat), 2*bootstrap_num-2)); % assuming large df

fprintf('\nComparing %s vs %s:\n', method_name, ref_method_name);


if p_value < 0.05
    fprintf('RESULT: The two methods yield significantly different results (p < 0.05)\n');
else
    fprintf('RESULT: No significant difference between the two methods (p ≥ 0.05)\n');
end

end