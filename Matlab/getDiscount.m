function discountnuovi=getDiscount(dates,discounts,datepag)
% Interpolates zero rates and computes discount factors for a set of payment dates.
%
% INPUTS:
%   dates       - Dates corresponding to market discount factors
%   discounts   - Discount factors at 'dates'
%   datepag     - Target payment dates
%
% OUTPUT:
%   discountnuovi - Interpolated discount factors at 'datepag'
%
%%

% Compute year fraction between today and target payment dates
delta=yearfrac(dates(1),datepag,3); % using ACT/365 convention

% Compute zero rates from discount factors
zrate=zeroRates(dates(2:end),discounts(2:end))/100;

% Interpolate zero rates at desired payment dates
ratenuovi=interp1(dates(2:end),zrate,datepag,'linear');

% Compute discount factors using the interpolated zero rates
discountnuovi=exp(-ratenuovi.*delta);
end
