import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm

from Utilities.boot import bootstrap, business_date_offset, getDiscount


class DiscountHandle:

    def __init__(self, file_path="MktData_CurveBootstrap.xls"):
        self.file_path = file_path
        self.today = pd.to_datetime('2023-02-02')

        # Load market data
        data = self.extract_data(file_path)
        self.dates_set = self._organize_dates(data)
        self.rates_set = self._organize_rates(data)

        # Bootstrap curve and calculate discounts
        self._bootstrap_and_calculate()
        self._save_to_latex()


    def extract_data(self, file_path):
        """Extract all market data from Excel file."""
        df = pd.read_excel(file_path, sheet_name="Sheet1")

        return {
            "settlement_date": df.iloc[6, 4],
            "depos_dates": df.iloc[9:15, 3].tolist(),
            "futures_dates": df.iloc[10:19, 16:18].values.tolist(),
            "swaps_dates": df.iloc[37:54, 3].tolist(),
            "depos_rates": self._convert_rates(df.iloc[9:15, 7:9].values),
            "futures_rates": self._convert_rates(df.iloc[26:35, 7:9].values),
            "swaps_rates": self._convert_rates(df.iloc[37:54, 7:9].values)
        }

    def _convert_rates(self, rates):
        """Convert rates to float and divide by 100."""
        return np.array(rates, dtype=np.float64) / 100

    def _organize_dates(self, data):
        """Organize dates into structured dictionary."""
        return {
            'settlement': data['settlement_date'],
            'depos': data['depos_dates'],
            'futures': data['futures_dates'],
            'swaps': data['swaps_dates']
        }

    def _organize_rates(self, data):
        """Organize rates into structured dictionary."""
        return {
            'depos': data['depos_rates'],
            'futures': data['futures_rates'],
            'swaps': data['swaps_rates']
        }

    def _bootstrap_and_calculate(self):
        """Bootstrap curve and calculate payment discounts."""
        dates, all_discounts = bootstrap(self.dates_set, self.rates_set)
        self.curve_dates = dates
        self.all_discounts = all_discounts

        # Calculate payment dates and discounts
        self.payment_dates = [business_date_offset(self.today, month_offset=6 * i) for i in range(0, 5)]
        df_expiry = getDiscount(dates, all_discounts, self.payment_dates)
        self.discounts = df_expiry[1:]

    def get_discounts_dataframe(self):
        """Get discount factors as DataFrame with corresponding dates."""
        return pd.DataFrame({
            'Date': self.payment_dates[1:],
            'Discount_Factor': self.discounts
        })

    def get_full_curve_dataframe(self):
        """Get the complete discount curve as DataFrame."""
        return pd.DataFrame({
            'Date': self.curve_dates,
            'Discount_Factor': self.all_discounts
        })

    def print_today(self):
        """Print today's date."""
        print(f"Today's date: {self.today}")

    def print_target_df(self):
        """Print the discount factors DataFrame."""
        df = self.get_discounts_dataframe()
        print("Discount Factors DataFrame:")
        print(df)

    def _save_to_latex(self):

        latex_table = self.get_discounts_dataframe().to_latex(
                index=False,
                float_format='{:.5f}'.format,
                bold_rows=True,
                escape=False,
                caption='Discount Factors (Semi-Annual)',
                label='tab:Discounts',
        )

        with open('discounts.tex', 'w') as f:
                f.write(latex_table)


class LoadData:
    def __init__(self):
        self.rating_categories = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default']

        # Extract the data
        empirical_data = loadmat('Data/empirical_joint_prob.mat')
        transition_data = loadmat('Data/transition_matrix.mat')
        self.empirical_joint_prob = empirical_data['empirical_joint_prob']
        self.transition_matrix = transition_data['transition_matrix']
        self.observation_num = 789683

        # Create DataFrames for easier manipulation
        self._create_dataframes()
        self.create_portfolio()
        self._save_as_latex()

    def _create_dataframes(self):
        # Convert numpy arrays to pandas DataFrames with labels
        n_ratings = len(self.rating_categories)
        categories = self.rating_categories[:n_ratings]

        # DataFrame for empirical joint probabilities
        self.empirical_joint_prob_df = pd.DataFrame(
            self.empirical_joint_prob,
            index=categories,
            columns=categories
        )

        # DataFrame for transition matrix - handle different dimensions
        transition_rows, transition_cols = self.transition_matrix.shape

        # Use appropriate number of categories for rows and columns
        row_categories = self.rating_categories[:transition_rows]
        col_categories = self.rating_categories[:transition_cols]

        self.transition_matrix_df = pd.DataFrame(
            self.transition_matrix,
            index=row_categories,
            columns=col_categories
        )

    def show_transition_matrix(self):
        print("\nTransition Matrix:")
        print(f"Shape: {self.transition_matrix.shape}")
        print(self.transition_matrix_df)

    def show_empirical_joint_prob(self):
        print("\nEmpirical Joint Probabilities:")
        print(f"Shape: {self.empirical_joint_prob.shape}")
        print(self.empirical_joint_prob_df)

    def get_empirical_joint_prob(self, as_dataframe=True):
        """Get empirical joint probabilities"""
        return self.empirical_joint_prob_df if as_dataframe else self.empirical_joint_prob

    def get_transition_matrix(self, as_dataframe=True):
        """Get transition matrix"""
        return self.transition_matrix_df if as_dataframe else self.transition_matrix

    def get_probability(self, from_rating, to_rating, matrix_type='empirical'):
        """Get specific transition probability"""
        if matrix_type == 'empirical':
            return self.empirical_joint_prob_df.loc[from_rating, to_rating]
        elif matrix_type == 'transition':
            return self.transition_matrix_df.loc[from_rating, to_rating]

    def get_default_transition_matrix(self):
        # Check if 'Default' column exists in transition matrix
        if 'Default' in self.transition_matrix_df.columns:
            return self.transition_matrix_df['Default']
        else:
            print("Warning: 'Default' column not found in transition matrix")
            return None


    def validate_matrices(self):
        # Check empirical joint probabilities validity
        emp_row_sums = self.empirical_joint_prob.sum(axis=1)
        sums = emp_row_sums.sum(axis=0)

        print(f"Empirical matrix row sums close to 1.0: {np.allclose(sums, 1.0, atol=1e-3)}")

        # Check transition matrix
        trans_row_sums = self.transition_matrix_df.sum(axis=1)
        print(f"Transition matrix row sums close to 1.0: {np.allclose(trans_row_sums, 1.0, atol=1e-3)}")

        print("\nRow sums:")
        print("Empirical matrix:")
        print(emp_row_sums)
        print("\nTransition matrix:")
        print(trans_row_sums)

    def get_joint_cdf_A(self):
        marginal_A = self.empirical_joint_prob.sum(axis=0)  # somma righe
        cdf_A = np.cumsum(marginal_A)
        return cdf_A

    def get_joint_cdf_BBB(self):
        marginal_BBB = self.empirical_joint_prob.sum(axis=1)  # somma colonne
        cdf_BBB = np.cumsum(marginal_BBB)
        return cdf_BBB

    def get_joint_threshold_A(self):
        cdf_A = self.get_joint_cdf_A()
        z_A = np.concatenate(([-np.inf], norm.ppf(cdf_A)))
        return z_A

    def get_joint_threshold_BBB(self):
            cdf_A = self.get_joint_cdf_BBB()
            z_BBB = np.concatenate(([-np.inf], norm.ppf(cdf_A)))
            return z_BBB

    def create_portfolio(self):
        # Portfolio parameters
        self.issuers_num_A = 50
        self.issuers_num_BBB = 50
        self.recovery_rate = 0.4
        self.face_value = 1

    def print_portfolio(self):
        print(f"Portfolio Configuration:")
        print(f"  - A-rated bonds: {self.issuers_num_A}")
        print(f"  - BBB-rated bonds: {self.issuers_num_BBB}")
        print(f"  - Recovery rate: {self.recovery_rate:.1%}")
        print(f"  - Face value: {self.face_value}")

    def _save_as_latex(self):
        latex_table = self.empirical_joint_prob_df.to_latex(
            index=False,
            float_format='{:.5f}'.format,
            bold_rows=True,
            escape=False,
            caption='Joint Probability Matrix for firms rated A and BBB',
            label='tab:Probability Matrix',
            position='H'
        )

        # Save to file
        with open('joint_prob.tex', 'w') as f:
            f.write(latex_table)

        latex_table2 = self.transition_matrix_df.to_latex(
            index=False,
            float_format='{:.5f}'.format,  # Format decimals
            bold_rows=True,  # Bold row headers
            escape=False,
            caption='Seven rating classes (i.e. AAA, AA, A, BBB, BB, B, CCC) prior to default (Yearly)',
            label='tab:Transition Matrix',
            position='H'  # Force position (requires float package)
        )

        # Save to file
        with open('tansition_matrix.tex', 'w') as f:
            f.write(latex_table2)


