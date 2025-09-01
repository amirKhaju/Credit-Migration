import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import pandas as pd


class CreditPortfolioVaR:
    """
    Credit Portfolio VaR Calculator using Vasicek single-factor model

    This class implements Monte Carlo simulation for credit portfolio Value-at-Risk
    calculation considering rating migrations and defaults.
    """

    def __init__(self, data, discounts):
        self.transition_matrix = data.get_transition_matrix(False)
        self.recovery_rate = data.recovery_rate
        self.discounts = discounts
        self.defaults = self.transition_matrix[:, -1]
        self.issuers_num_BBB = data.issuers_num_BBB
        self.issuers_num_A = data.issuers_num_A

        # Calculate forward prices and thresholds
        self.forward_prices = self._calculate_forward_prices()
        self.thresholds = self._calculate_thresholds()  # Now returns dict
        self.initial_portfolio_value = self._compute_initial_portfolio_value()

    def _calculate_thresholds(self) -> dict:
        """Calculate rating migration thresholds using inverse normal CDF"""
        thresholds = {}
        probs_A = self.transition_matrix[2, :]  # A rating row
        cdf_A = np.cumsum(np.flip(probs_A))
        thresholds['A'] = stats.norm.ppf(cdf_A)

        # Calculate barriers for BBB-rated bonds (index 3)
        probs_BBB = self.transition_matrix[3, :]  # BBB rating row
        cdf_BBB = np.cumsum(np.flip(probs_BBB))
        thresholds['BBB'] = stats.norm.ppf(cdf_BBB)

        return thresholds

    def _calculate_forward_prices(self) -> np.ndarray:
        """Calculate forward bond prices for different rating states"""
        default_case = self.recovery_rate * self.discounts[0]
        fwd_prices = (self.discounts[3] / self.discounts[1] * (1 - self.defaults) +
                      self.discounts[2] / self.discounts[1] * self.defaults * self.recovery_rate)

        forward_prices = np.append(fwd_prices, default_case)
        return forward_prices

    def _compute_initial_portfolio_value(self) -> float:
        """
        Compute initial portfolio value accounting for credit risk and rating transitions
        """
        face_value = 1.0

        # Extract 1-year default probabilities
        pd_1y_A = self.defaults[2]  # A rating (index 2)
        pd_1y_BBB = self.defaults[3]  # BBB rating (index 3)
        pd_1y = self.defaults  # All ratings

        frwdisc1y2y = self.discounts[3] / self.discounts[1]  # B(0;1,2)=B(0,2)/B(0,1)
        frwdisc1y_1y6m = self.discounts[2] / self.discounts[1]  # B(0;1,3/2) = B(0,3/2)/B(0,1)
        frwdis6mesi = self.discounts[0]  # B(0;0,1/2)=B(0,1/2)/B(0,0)

        # Compute DF(1y→2y | x) where x can be AAA, AA, A, BBB, BB, B, CCC
        df_1y2ydef = ((1 - pd_1y) * frwdisc1y2y +
                      pd_1y * frwdisc1y_1y6m * self.recovery_rate)

        survival_probs_A = self.transition_matrix[2, :-1]  # A rating transitions (exclude default)
        df_survival_A = np.dot(survival_probs_A, self.discounts[1] * df_1y2ydef)
        df_default_A = self.transition_matrix[2, -1] * self.discounts[0] * self.recovery_rate
        df_2y_def_A = df_survival_A + df_default_A

        # For BBB-rated bonds
        survival_probs_BBB = self.transition_matrix[3, :-1]  # BBB rating transitions (exclude default)
        df_survival_BBB = np.dot(survival_probs_BBB, self.discounts[1] * df_1y2ydef)
        df_default_BBB = self.transition_matrix[3, -1] * self.discounts[0] * self.recovery_rate

        # Total expected discount factor at 2 years from initial rating "BBB"
        df_2y_def_BBB = df_survival_BBB + df_default_BBB

        # Current mark-to-market of a single 2y bond, accounting for credit risk and transitions
        bond_mtm_A = df_2y_def_A * face_value
        bond_mtm_BBB = df_2y_def_BBB * face_value

        # Mark-to-market of the entire portfolio (50 issuers A and 50 issuers BBB)
        ptf_mtm = bond_mtm_A * self.issuers_num_A + bond_mtm_BBB * self.issuers_num_BBB

        return ptf_mtm

    def _calculate_bond_values(self, firm_values: np.ndarray, barriers: np.ndarray,
                               forward_prices: np.ndarray, n_simulations: int,
                               n_issuers: int) -> Tuple[np.ndarray, np.ndarray]:
        # Create edges for digitization
        edges = np.concatenate(([-np.inf], barriers))

        # Assign rating states
        states = 9 - np.digitize(firm_values, edges)

        # Count ratings for each simulation
        counts = np.zeros((n_simulations, 8), dtype=int)
        for ii in range(1, 9):  # 1 = AAA, 8 = Default
            counts[:, ii - 1] = np.sum(states == ii, axis=1)

        # Calculate bond prices using your friend's formula
        frwdis6mesi = self.discounts[0]  # B(0;0,1/2)

        price1y = ((counts[:, :7] @ forward_prices[:7] +
                    counts[:, 7] * self.recovery_rate * frwdis6mesi) / n_issuers)

        # Average ratings across simulations
        avg_ratings = np.mean(counts, axis=0)

        return price1y, avg_ratings

    def _generate_random_variables(self, n_simulations: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random variables for Monte Carlo simulation"""
        np.random.seed(seed)
        epsilon_A = np.random.randn(n_simulations, 50)
        epsilon_BBB = np.random.randn(n_simulations, 50)
        y = np.random.randn(n_simulations, 1)
        return epsilon_A, epsilon_BBB, y

    def _vasicek_firm_values(self, epsilon_A: np.ndarray, epsilon_BBB: np.ndarray,
                             y: np.ndarray, rhoA: float, rhoBBB: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate firm values using Vasicek single-factor model"""
        v_A = np.sqrt(rhoA) * y + np.sqrt(1 - rhoA) * epsilon_A
        v_BBB = np.sqrt(rhoBBB) * y + np.sqrt(1 - rhoBBB) * epsilon_BBB
        return v_A, v_BBB

    def _calculate_initial_bond_mtm(self, rating_index: int) -> float:
        """Calculate initial mark-to-market value for bonds of a given rating"""
        ratings_intermediate = [0, 1, 2, 3, 4, 5, 6]  # AAA to CCC indices

        # Survival component
        df_survival = np.dot(self.transition_matrix[rating_index, ratings_intermediate],
                             self.discounts[1] * self.forward_prices[:7])

        # Default component
        df_default = (self.transition_matrix[rating_index, -1] *
                      self.discounts[0] * self.recovery_rate)

        return df_survival + df_default

    def _calculate_portfolio_losses(self, price1y_A: np.ndarray, price1y_BBB: np.ndarray) -> np.ndarray:
        """Calculate portfolio losses for each simulation scenario"""
        # Get initial bond MTM values
        bond_mtm_A = self._calculate_initial_bond_mtm(rating_index=2)  # A rating
        bond_mtm_BBB = self._calculate_initial_bond_mtm(rating_index=3)  # BBB rating

        # Calculate losses per bond type
        lossexd_A = bond_mtm_A / self.discounts[1] - price1y_A
        lossexd_BBB = bond_mtm_BBB / self.discounts[1] - price1y_BBB

        # Total portfolio losses
        portfolio_losses = lossexd_A * self.issuers_num_A + lossexd_BBB * self.issuers_num_BBB

        return portfolio_losses

    def _calculate_var(self, portfolio_losses: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk at given confidence level"""
        return np.percentile(portfolio_losses, confidence_level * 100)

    def _format_results(self, var_value: float, portfolio_losses: np.ndarray,
                        ratings_A: np.ndarray, ratings_BBB: np.ndarray,
                        rhoA: float, rhoBBB: float, n_simulations: int) -> dict:
        """Format simulation results into a dictionary"""
        return {
            'var': var_value,
            'portfolio_losses': portfolio_losses,
            'A_probabilities': ratings_A / (n_simulations * self.issuers_num_A),
            'BBB_probabilities': ratings_BBB / (n_simulations * self.issuers_num_BBB),
            'correlations': [rhoA, rhoBBB]
        }

    def simulate_portfolio_var(self, rhoA: float, rhoBBB: float, n_simulations: int = 1000000,
                               confidence_level: float = 0.999, seed: int = 0) -> dict:
        """
        Main Monte Carlo simulation for portfolio VaR calculation
        Returns: Dictionary containing VaR and simulation results
        """
        epsilon_A, epsilon_BBB, y = self._generate_random_variables(n_simulations, seed)

        #  calculate firm values using Vasicek model
        v_A, v_BBB = self._vasicek_firm_values(epsilon_A, epsilon_BBB, y, rhoA, rhoBBB)

        # Get rating migration barriers
        barriers_A = self.thresholds['A']
        barriers_BBB = self.thresholds['BBB']

        # Calculate bond prices at 1-year horizon by doing mc simulation
        price1y_A, ratings_A = self._calculate_bond_values(
            v_A, barriers_A, self.forward_prices, n_simulations, self.issuers_num_A)
        price1y_BBB, ratings_BBB = self._calculate_bond_values(
            v_BBB, barriers_BBB, self.forward_prices, n_simulations, self.issuers_num_BBB)

        # Calculate portfolio losses
        portfolio_losses = self._calculate_portfolio_losses(price1y_A, price1y_BBB)

        # Calculate VaR
        var_value = self._calculate_var(portfolio_losses, confidence_level)

        # Format and return results
        return self._format_results(var_value, portfolio_losses, ratings_A, ratings_BBB,
                                    rhoA, rhoBBB, n_simulations)

    def plot_portfolio_losses(self, portfolio_losses: np.ndarray, var_value: float,
                              confidence_level: float, title: str = "Portfolio Loss Distribution"):
        plt.figure(figsize=(12, 8))

        # Plot histogram
        plt.hist(portfolio_losses, bins=100, density=True, alpha=0.7,
                 color='lightblue', edgecolor='black')

        plt.axvline(var_value, color='red', linestyle='--', linewidth=2,
                    label=f'VaR ({confidence_level * 100}%): {var_value:.6f}')

        plt.xlabel('Portfolio Losses')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        stats_text = f'Mean Loss: {np.mean(portfolio_losses):.6f}\n'
        stats_text += f'Std Dev: {np.std(portfolio_losses):.6f}\n'
        stats_text += f'Max Loss: {np.max(portfolio_losses):.6f}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(title, dpi=300, bbox_inches='tight')
        plt.show()

    def run_analysis(self, rhoA, rhoBBB, n_simulations: int = 1000000,
                     confidence_level: float = 0.999):
        print("Running Credit Portfolio VaR Analysis...")
        print("=" * 50)

        results = self.simulate_portfolio_var(rhoA, rhoBBB, n_simulations, confidence_level)
        print(f"Value at Risk (VaR): {results['var']:.8f}")
        return results

    def print_results_summary(self, results: dict, title: str = "VaR Analysis Results") -> None:
        """Print a formatted summary of VaR analysis results"""
        print(f"\n{title}")
        print("-" * len(title))
        print(f"VaR (99.9%):              ${results['var']:>12,.5f}")
        print(f"Mean portfolio loss:      ${np.mean(results['portfolio_losses']):>12,.5f}")

        # VaR as percentage of portfolio
        var_pct = (results['var'] / self.initial_portfolio_value) * 100
        print(f"VaR as % of portfolio:    {var_pct:>12.5f}%")