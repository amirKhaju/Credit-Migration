# Credit Portfolio VaR with Rating Migrations

This project implements a structural credit portfolio model to calibrate correlation parameters and calculate Credit Value-at-Risk (VaR) based on rating migrations data.
Project Overview
The project addresses the sensitivity of Credit Portfolio VaR to the correlation parameter (ρ) between standardized asset returns, which is an unobservable parameter that remains challenging to calibrate from historical data.
Key components:

Calibration of correlation parameter ρ using historical joint rating migration data
Implementation of a single-factor Markov Chain model for Credit Portfolio VaR calculation
Analysis of portfolio risk under different correlation assumptions
Comparison of results between constant correlation and rating-dependent correlation models

Features

Calibration of asset correlation from joint rating migration probabilities
Monte Carlo simulation of credit portfolio with multiple rating classes
Implementation of the Vasicek single-factor model for defaults and migrations
VaR calculation for portfolios of zero-coupon bonds with different rating profiles
Analysis of correlation impacts on portfolio risk assessment

Technologies

MATLAB and Python implementations
Statistical optimization for correlation parameter calibration
Monte Carlo simulation for credit portfolio analysis

References

Gupton, G. M., Finger C. C. and Bhatia, M. CreditMetrics - Technical Document J.P. Morgan, New York 1997.
Frye, Jon. "Correlation and asset correlation in the structural portfolio model." The Journal of Credit Risk 4.2 (2008): 75-96.
Basel Committee on Banking Supervision, An Explanatory Note on the Basel II IRB Risk Weight functions, July 2005.

Academic Context
Part of Financial Engineering coursework (AY 2024-2025), focusing on quantitative risk management and credit portfolio modeling.
