
function zRates = zeroRates(t0, dates, discounts)
% zeroRates - Calculate zero rates from discount factors
%
% Inputs:
%   t0        - Initial date (datetime)
%   dates     - Vector of future dates (datetime array)
%   discounts - Vector of discount factors (numeric array)
%
% Outputs:
%   zRates    - Vector of zero rates (numeric array)
ttm = yearfrac(t0, dates, 3);      
zRates = -log(discounts) ./ ttm;    
end
