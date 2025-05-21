function correlation = calculateBaselCorrelation(PD)
annualSales =50;
% Basel II correlation formula for corporates
term1 = 0.12 * (1 - exp(-50 * PD)) / (1 - exp(-50));
term2 = 0.24 * (1 - (1 - exp(-50 * PD)) / (1 - exp(-50)));
baseCorrelation = term1 + term2;

% Apply size adjustment for SMEs
S = min(max(5, annualSales), 50);
sizeAdjustment = 0.04 * (1 - (S - 5) / 45);
correlation = max(0, baseCorrelation - sizeAdjustment);

end