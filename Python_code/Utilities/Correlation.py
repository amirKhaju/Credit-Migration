from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, t


class CorrelationCalibrator:
    """
    A class to calibrate correlation parameters using various loss functions
    and bootstrap methods for confidence intervals and model validation.
    """

    def __init__(self, data):
        z_A = data.get_joint_threshold_A()
        z_BBB = data.get_joint_threshold_BBB()
        self.observed_joint_probs = data.get_empirical_joint_prob(False)
        self.z_BBB = np.clip(z_BBB, -20, 20)
        self.z_A = np.clip(z_A, -20, 20)
        self.rho_min = 0.00001
        self.rho_max = 0.99999
        self.rating_labels = data.rating_categories

    def calculate_theoretical_joint_probs(self, rho) -> np.ndarray:
        """
        Calculate theoretical joint probabilities using bivariate normal distribution.
        Returns:8x8 matrix of theoretical joint probabilities
        """
        # Handle both scalar and array inputs (extract scalar if needed)
        if isinstance(rho, np.ndarray):
            rho = rho.item() if rho.size == 1 else rho[0]
        elif isinstance(rho, (list, tuple)):
            rho = rho[0]

        # Create meshgrid for all combinations
        I, J = np.meshgrid(range(8), range(8), indexing='ij')

        z_BBB_i = self.z_BBB[I]
        z_BBB_i_plus_1 = self.z_BBB[I + 1]
        z_A_j = self.z_A[J]
        z_A_j_plus_1 = self.z_A[J + 1]

        # Correlation matrix and mean
        cov_matrix = np.array([[1, rho], [rho, 1]])
        mean = np.array([0, 0])

        # Create multivariate normal distribution
        mvn = multivariate_normal(mean=mean, cov=cov_matrix)
        point_11 = mvn.cdf(np.column_stack([z_BBB_i_plus_1.ravel(), z_A_j_plus_1.ravel()]))
        point_10 = mvn.cdf(np.column_stack([z_BBB_i_plus_1.ravel(), z_A_j.ravel()]))
        point_01 = mvn.cdf(np.column_stack([z_BBB_i.ravel(), z_A_j_plus_1.ravel()]))
        point_00 = mvn.cdf(np.column_stack([z_BBB_i.ravel(), z_A_j.ravel()]))

        joint_probs = (point_11 - point_10 - point_01 + point_00).reshape(8, 8)
        return joint_probs

    def _loss_function(self, rho: float, observed_probs, mode: str) -> float:
        """
        Calculate loss function with custom observed probabilities.
        This allows for bootstrap analysis with different probability matrices.
        """
        # Calculate theoretical probabilities
        theoretical_probs = self.calculate_theoretical_joint_probs(rho)

        # Small epsilon to avoid numerical issues
        eps = 1e-10

        if mode == 'MSE':
            loss = np.sum((theoretical_probs - observed_probs) ** 2)

        elif mode == 'MAE':
            loss = np.sum(np.abs(theoretical_probs - observed_probs))

        elif mode == 'likelihood':
            mask = observed_probs > 0
            log_likelihood = np.sum(observed_probs[mask] *
                                    np.log(theoretical_probs[mask] + eps))
            loss = -log_likelihood

        elif mode == 'KL':
            # Kullback-Leibler divergence
            P = observed_probs.flatten()
            Q = np.maximum(theoretical_probs.flatten(), eps)
            mask = P > 0
            loss = np.sum(P[mask] * np.log(P[mask] / Q[mask]))

        elif mode == 'JSD':
            # Jensen-Shannon divergence
            P = observed_probs.flatten()
            Q = np.maximum(theoretical_probs.flatten(), eps)
            M = 0.5 * (P + Q)
            mask = M > 0
            D1 = np.sum(P[mask] * np.log((P[mask] + eps) / (M[mask] + eps)))
            D2 = np.sum(Q[mask] * np.log((Q[mask] + eps) / (M[mask] + eps)))
            loss = 0.5 * (D1 + D2)

        elif mode == 'weighted_MSE':
            i_idx, j_idx = np.meshgrid(np.arange(1, 9), np.arange(1, 9), indexing='ij')
            weights = i_idx + j_idx
            squared_errors = (theoretical_probs - observed_probs) ** 2
            weighted_errors = weights * squared_errors
            loss = np.sum(weighted_errors)

        elif mode == 'weighted_MAE':
            i_idx, j_idx = np.meshgrid(np.arange(1, 9), np.arange(1, 9), indexing='ij')
            weights = i_idx + j_idx
            errors = np.abs(theoretical_probs - observed_probs)
            weighted_errors = weights * errors
            loss = np.sum(weighted_errors)

        else:
            raise ValueError(f"Unsupported loss function: {mode}")

        return loss

    def _grid_search(self, mode: str, observed_probs,num_grid_points: int = 20) -> Tuple[float, float]:
        """
        Perform grid search to find initial rho value.
        Returns: Best grid rho and corresponding loss value
        """
        grid_rhos = np.linspace(self.rho_min + 0.01, self.rho_max - 0.01, num_grid_points)
        # Fixed: removed incorrect observed_joint_probs parameter
        grid_losses = np.array([self._loss_function(rho, observed_probs ,mode) for rho in grid_rhos])

        min_idx = np.argmin(grid_losses)
        best_grid_rho = grid_rhos[min_idx]
        min_grid_loss = grid_losses[min_idx]

        return best_grid_rho, min_grid_loss

    def _calibrate_rho(self, mode: str, observed_probs) -> Tuple[float, float]:
        """
        Returns Optimized rho and loss value
        """
        best_grid_rho, min_grid_loss = self._grid_search(mode,self.observed_joint_probs)

        objective_func = lambda rho: self._loss_function(rho, observed_probs,mode)
        x0 = np.array([best_grid_rho])
        bounds = [(self.rho_min, self.rho_max)]
        result = minimize(fun=objective_func, x0=x0, bounds=bounds, tol=1e-10)
        calibrated_rho = result.x[0]
        loss_value = result.fun

        # Check if optimization improved on grid search
        if loss_value > min_grid_loss:
            calibrated_rho = best_grid_rho
            loss_value = min_grid_loss

        return calibrated_rho, loss_value

    def _calibrate_rho_multi_grid(self, mode: str, observed_probs) -> Tuple[float, float]:
        """
        Returns Optimized rho and loss value
        """
        # Replace grid search with multiple starting points
        starting_points = np.linspace(self.rho_min + 0.01, self.rho_max - 0.01, 20)

        best_rho = None
        best_loss = np.inf

        objective_func = lambda rho: self._loss_function(rho, observed_probs, mode)
        bounds = [(self.rho_min, self.rho_max)]

        # Try optimization from each starting point
        for x0_val in starting_points:
            x0 = np.array([x0_val])
            result = minimize(fun=objective_func, x0=x0, bounds=bounds, tol=1e-10)

            if result.success and result.fun < best_loss:
                best_rho = result.x[0]
                best_loss = result.fun

        # Fallback to first starting point if all failed
        if best_rho is None:
            x0 = np.array([starting_points[0]])
            result = minimize(fun=objective_func, x0=x0, bounds=bounds, tol=1e-10)
            best_rho = result.x[0]
            best_loss = result.fun

        return best_rho, best_loss



    def _generate_bootstrap_sample(self, total_size: int) -> np.ndarray:
        """
        Generate a single bootstrap sample using multinomial distribution.
        Returns: Bootstrap probability matrix
        """
        flat_probs = self.observed_joint_probs.flatten()
        bootstrap_counts = np.random.multinomial(total_size, flat_probs)
        bootstrap_probs = bootstrap_counts.reshape(self.observed_joint_probs.shape) / total_size
        return bootstrap_probs

    def _perform_bootstrap_analysis(self, mode: str, n_bootstrap: int, total_size: int) -> np.ndarray:
        """
        Perform bootstrap analysis to get distribution of rho estimates.
        Returns: Array of bootstrap rho estimates
        """
        print("Performing bootstrap analysis...")
        rhos = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            # Generate bootstrap sample
            bootstrap_probs = self._generate_bootstrap_sample(total_size)
            calibrated_rho, loss_value = self._calibrate_rho(mode, bootstrap_probs)
            rhos[i] = calibrated_rho
        return rhos

    def _calculate_confidence_interval(self, bootstrap_rhos: np.ndarray,
                                       confidence_level: float) -> Tuple[float, float]:
        """
        Calculate confidence interval from bootstrap samples.
        Returns: Lower and upper confidence interval bounds
        """
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ci_lower = np.percentile(bootstrap_rhos, lower_percentile)
        ci_upper = np.percentile(bootstrap_rhos, upper_percentile)

        return ci_lower, ci_upper

    def _perform_hypothesis_test(self, bootstrap_rhos: np.ndarray, method_mean,
                                 confidence_level, null_value: float = 0) -> float:
        """
        Perform hypothesis test using bootstrap distribution.
        Returns: p-value of the test
        """
        n_bootstrap = len(bootstrap_rhos)
        if method_mean > null_value:
            p_value = 2 * np.sum(bootstrap_rhos <= null_value) / n_bootstrap
        else:
            p_value = 2 * np.sum(bootstrap_rhos >= null_value) / n_bootstrap

        return p_value

    def _print_results(self, calibrated_rho, loss_value, ci_lower,
                       ci_upper, method_se, p_value, confidence_level, mode):
        # Print calibration results.
        print(f"Calibrated rho: {calibrated_rho:.6f} with loss: {loss_value:.8e}")
        print(f"{confidence_level * 100:.1f}% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"Bootstrap std error: {method_se:.6f}")
        print(f"Bootstrap p-value: {p_value:.6f}")

        if p_value < (1 - confidence_level):
            print(f"Reject H0 (rho=0) at {confidence_level * 100:.1f}% confidence level (p-value method)")
        else:
            print(f"Fail to reject H0 (rho=0) at {confidence_level * 100:.1f}% confidence level (p-value method)")

    def analyze_method(self, mode, calibration_df,n_bootstrap: int = 400,
                       confidence_level: float = 0.95, total_size: int = 789683) -> Dict:
        """
        run this function if you want to focus only in one method
        """
        print("-"*50)
        print(f"\nAnalysing with {mode} loss function...")
        row = calibration_df[calibration_df['Method'] == mode]
        calibrated_rho = row['Calibrated_Rho'].values[0]
        loss_value = row['Loss_Value'].values[0]

        # Grid search for initial value
        #calibrated_rho, loss_value = self._calibrate_rho(mode,self.observed_joint_probs)
        bootstrap_rhos = self._perform_bootstrap_analysis(mode, n_bootstrap, total_size)
        method_mean = np.mean(bootstrap_rhos)
        method_se = np.std(bootstrap_rhos, ddof=1)
        ci_lower, ci_upper = self._calculate_confidence_interval(bootstrap_rhos, confidence_level)
        p_value = self._perform_hypothesis_test(bootstrap_rhos, method_mean, confidence_level)
        self._print_results(calibrated_rho, loss_value, ci_lower, ci_upper,
                            method_se, p_value, confidence_level, mode)

        return {
            'calibrated_rho': calibrated_rho,
            'confidence_interval': [ci_lower, ci_upper],
            'method_mean': method_mean,
            'method_se': method_se,
            'loss_value': loss_value,
            'p_value': p_value,
            'bootstrap_samples': bootstrap_rhos
        }

    def compare_methods_with_reference(self, method_name: str, ref_method_name: str,
                                       bootstrap_rhos_method, bootstrap_rhos_ref ) -> None:

        # Calculate paired differences
        paired_diffs = bootstrap_rhos_method - bootstrap_rhos_ref
        diff_mean = np.mean(paired_diffs)
        diff_se = np.std(paired_diffs, ddof=1)

        # Calculate t-statistic and p-value
        t_stat = diff_mean / diff_se
        df = len(paired_diffs) - 1  # Correct degrees of freedom
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))

        print(f"\nComparing {method_name} vs {ref_method_name}:")
        print(f"Difference in means: {diff_mean:.6f}")
        print(f"Standard error of difference: {diff_se:.6f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")

        if p_value < 0.05:
            print("RESULT: The two methods yield significantly different results (p < 0.05)")
        else:
            print("RESULT: No significant difference between the two methods (p ≥ 0.05)")

    def _analyze_all_methods(self,modes, calibration_df,n_bootstrap=100, confidence_level=0.95) -> Dict[str, Dict]:
        """
        Analyze all specified methods.
        Returns:  Dictionary with method names as keys and calibration results as values
        """
        all_results = {}

        for mode in modes:
            result = self.analyze_method(mode, calibration_df,n_bootstrap=100, confidence_level=0.95)
            all_results[mode] = result

        return all_results

    def calibrate_all_methods(self, modes):
        return pd.DataFrame([
            {'Method': mode, 'Calibrated_Rho': rho, 'Loss_Value': loss}
            for mode in modes
            for rho, loss in [self._calibrate_rho_multi_grid(mode, self.observed_joint_probs)]
        ])

    def _create_results_table(self, all_results: Dict[str, Dict]) -> List[Dict]:
        """
        Create a list of result dictionaries for DataFrame creation.
        Returns: List of dictionaries with formatted results for each method
        """
        results = []

        for mode, result in all_results.items():
            results.append({
                'Method': mode,
                'Calibrated_Rho': result['calibrated_rho'],
                'Bootstrap_Mean': result['method_mean'],
                'Bootstrap_SE': result['method_se'],
                'CI_Lower': result['confidence_interval'][0],
                'CI_Upper': result['confidence_interval'][1],
                'Loss_Value': result['loss_value']
            })

        return results

    def _compare_all_methods_with_reference(self, modes: List[str], all_results: Dict[str, Dict],
                                            reference_method: str = 'weighted_MSE', **kwargs) -> None:
        """
        Compare all methods against a reference method.
        **kwargs : dict
            Additional arguments (used to extract n_bootstrap)
        """
        if reference_method not in modes:
            print(f"\nReference method '{reference_method}' not found in modes. Skipping comparisons.")
            return

        ref_result = all_results[reference_method]

        print(f"\nComparing all methods with {reference_method}:")
        print("=" * (25 + len(reference_method)))

        for mode in modes:
            if mode != reference_method:
                method_result = all_results[mode]
                self.compare_methods_with_reference(
                    mode, reference_method,
                    method_result['bootstrap_samples'], ref_result['bootstrap_samples'])

    def run_comprehensive_analysis(self, modes, calibration_df,n_bootstrap=100, confidence_level=0.95) -> pd.DataFrame:
        """
        Calibrate correlation using multiple loss functions and compare results.
        Returns:
        --------
        Tuple[pd.DataFrame, Dict[str, Dict]]
            - DataFrame with comparison results for all methods
            - Dictionary with detailed results for each method
        """
        # Analyze all methods
        all_results = self._analyze_all_methods(modes, calibration_df,n_bootstrap=100, confidence_level=0.95)

        # Compare methods with reference method
        self._compare_all_methods_with_reference(modes, all_results)

        # Create results table
        results_list = self._create_results_table(all_results)
        results_df = pd.DataFrame(results_list)
        self._save_as_latex(results_df)

        return results_df, all_results

    def plot_objective_function(self, mode: str, calibrated_rho: float = None,
                                rho_range: Tuple[float, float] = (0.001, 0.1),
                                n_points: int = 100, figsize: Tuple[int, int] = (10, 6)):
        # Generate rho values
        rho_values = np.linspace(rho_range[0], rho_range[1], n_points)

        # Calculate loss for each rho value
        print(f'Calculating loss values for {mode} plotting...')
        loss_values = np.array([self._loss_function(rho, self.observed_joint_probs, mode)
                                for rho in rho_values])

        # Find minimum if not provided
        if calibrated_rho is None:
            min_idx = np.argmin(loss_values)
            calibrated_rho = rho_values[min_idx]

        min_loss = self._loss_function(calibrated_rho, self.observed_joint_probs, mode)

        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(rho_values, loss_values, 'b-', linewidth=2, label='Loss Function')
        plt.plot(calibrated_rho, min_loss, 'ro', markersize=10,
                 markerfacecolor='r', label='Minimum')

        plt.xlabel('Correlation Parameter (ρ)', fontsize=12)
        plt.ylabel(f'Loss Value ({mode})', fontsize=12)
        plt.title(f'Objective Function for {mode} Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text annotation
        plt.annotate(f'Min ρ = {calibrated_rho:.4f}\nMin Loss = {min_loss:.6e}',
                     xy=(calibrated_rho, min_loss),
                     xytext=(calibrated_rho + 0.01, min_loss),
                     fontsize=10, verticalalignment='bottom',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_multiple_objective_functions(self, modes: List[str],
                                          rho_range: Tuple[float, float] = (0.001, 0.1),
                                          n_points: int = 100,
                                          figsize: Tuple[int, int] = (12, 8)):
        """
        Plot multiple objective functions on the same figure for comparison.
        """
        rho_values = np.linspace(rho_range[0], rho_range[1], n_points)

        plt.figure(figsize=figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))

        for i, mode in enumerate(modes):
            loss_values = np.array([self._loss_function(rho, self.observed_joint_probs, mode)
                                    for rho in rho_values])

            # Normalize loss values for better comparison
            normalized_loss = (loss_values - np.min(loss_values)) / (np.max(loss_values) - np.min(loss_values))

            plt.plot(rho_values, normalized_loss, color=colors[i],
                     linewidth=2, label=f'{mode}')

            # Mark minimum
            min_idx = np.argmin(loss_values)
            min_rho = rho_values[min_idx]
            plt.plot(min_rho, normalized_loss[min_idx], 'o',
                     color=colors[i], markersize=8, markerfacecolor=colors[i])

        plt.xlabel('Correlation Parameter (ρ)', fontsize=12)
        plt.ylabel('Normalized Loss Value', fontsize=12)
        plt.title('Comparison of Loss Functions', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Objective functions", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_joint_distribution(self, rho: float, grid_range: Tuple[float, float] = (-4, 4),
                                     grid_points: int = 100, figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize the joint bivariate normal distribution with rating thresholds.
        """
        # Create grid for plotting
        x = np.linspace(grid_range[0], grid_range[1], grid_points)
        y = np.linspace(grid_range[0], grid_range[1], grid_points)
        X, Y = np.meshgrid(x, y)

        # Calculate bivariate normal PDF
        pos = np.dstack((X, Y))
        rv = multivariate_normal([0, 0], [[1, rho], [rho, 1]])
        Z = rv.pdf(pos)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 2D contour plot with thresholds
        contour = axes[0].contour(X, Y, Z, levels=15, linewidths=1)
        axes[0].clabel(contour, inline=True, fontsize=8)

        # Add threshold lines
        for i, threshold in enumerate(self.z_BBB):
            axes[0].axvline(x=threshold, color='blue', linestyle='--',
                            linewidth=1, alpha=0.7)

        for i, threshold in enumerate(self.z_A):
            axes[0].axhline(y=threshold, color='red', linestyle='--',
                            linewidth=1, alpha=0.7)

        # Add region labels
        for i in range(7):
            for j in range(7):
                if i < len(self.z_BBB) - 1 and j < len(self.z_A) - 1:
                    midpoint_x = (self.z_BBB[i] + self.z_BBB[i + 1]) / 2
                    midpoint_y = (self.z_A[j] + self.z_A[j + 1]) / 2

                    if (grid_range[0] < midpoint_x < grid_range[1] and
                            grid_range[0] < midpoint_y < grid_range[1]):
                        axes[0].text(midpoint_x, midpoint_y,
                                     f'{self.rating_labels[i]}-{self.rating_labels[j]}',
                                     ha='center', va='center', fontsize=6,
                                     bbox=dict(boxstyle="round,pad=0.2",
                                               facecolor="white", alpha=0.7))

        axes[0].set_xlabel('BBB-rated Firm Asset Return')
        axes[0].set_ylabel('A-rated Firm Asset Return')
        axes[0].set_title(f'Joint Distribution with Rating Thresholds (ρ = {rho:.5f})')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(grid_range)
        axes[0].set_ylim(grid_range)

        # 3D surface plot
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax_3d.set_xlabel('BBB-rated Firm Asset Return')
        ax_3d.set_ylabel('A-rated Firm Asset Return')
        ax_3d.set_zlabel('Probability Density')
        ax_3d.set_title(f'3D Bivariate Normal Density (ρ = {rho:.3f})')
        ax_3d.view_init(30, 30)

        plt.tight_layout()
        plt.savefig("Joint distribution plot", dpi=300, bbox_inches='tight')
        plt.show()

    def _save_as_latex(self, results_df):
        latex_table = results_df.to_latex(
            index=False,
            float_format='{:.5f}'.format,  # Format decimals
            bold_rows=True,  # Bold row headers
            escape=False,
            caption=' Correlation result full table',
            label='tab:correlation result',
            position='H'  # Force position (requires float package)
        )
        # Save to file
        with open('corr-result.tex', 'w') as f:
            f.write(latex_table)



def rho_function_Basel(prob_def):
    S = 50  # 50 million annual sales
    k = 50
    return (
        0.12 * (1 - np.exp(-k * prob_def)) / (1 - np.exp(-k))
        + 0.24 * (1 - (1 - np.exp(-k * prob_def)) / (1 - np.exp(-k)))
        - 0.04 * (1 - (S - 5) / 45)
    )

