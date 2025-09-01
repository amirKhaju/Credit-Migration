function zRates = zeroRates(dates, discounts)
%   Custom function to compute zero rates from discounts
%   INPUT
%   dates: dates' vector of expiries of the instrument of interest
%   discounts: discounts' vector with respect to each corresponding expiry
    
    % manually insert settlement date to ensure its presence
    settlements = datetime(2023,2,2);
    set=datenum(settlements);
    delta = yearfrac(set, dates(1:length(discounts)),3);
    % zero rate computation
    zRates = (-log(discounts)./delta).*100;
end