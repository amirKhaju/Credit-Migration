function [datesswap, rateswap]=swaperate(datesSet, ratesSet)
%% SWAP RATE INTERPOLATION FUNCTION
% This function computes swap rates by extracting market swap data, adjusting
% payment dates to business days, and performing interpolation to generate
% swap rates up to a specified final date.
%
% INPUTS:
%   datesSet  - Structure containing key dates for financial instruments.
%               - datesSet.swaps: Vector of swap payment dates.
%
%   ratesSet  - Structure containing bid-ask rates for swaps.
%               - ratesSet.swaps: Matrix containing bid and ask swap rates.
%
% OUTPUTS:
%   datesswap - Vector of adjusted swap dates, ensuring they fall on business days.
%   rateswap  - Vector of interpolated swap rates using spline interpolation.
%
%%
% Compute the mid rate between bid and ask swap rates
ratemid=(ratesSet.swaps(:,1)+ratesSet.swaps(:,2))/2;

% Assign the first 11 swap dates directly from the excel data
datesswap(1:11)=datesSet.swaps(1:11);

% Define settlement date and final date for swap rates
settlements = datetime(2035,2,2);
finaldate = datetime(2073,2,2);

% Generate an array of annual dates between the settlement and the final date
ipodate = settlements + calyears(1:years(finaldate - settlements));


%%  Adjust dates to ensure they fall on business days
ipodatenew = datenum(ipodate);  % Convert datetime to numeric

% Adjust non-business days
ipodatenew(~isbusday(ipodatenew)) = busdate(ipodatenew(~isbusday(ipodatenew)));

% assign the correct business days to the swap dates array
datesswap(12:49)=ipodatenew;

% linear interpolation to obtain swap rates
rateswap=interp1(datesSet.swaps,ratemid,datesswap,"spline");

end