function [dates, discount] = bootstrap(datesSet, ratesSet)
% This function performs bootstrapping to derive discount factors and zero rates
% from a given set of market data including deposit rates, futures rates, and swap rates.
% 
% Inputs:
% - datesSet: A structure containing the settlement date and dates for deposits, futures, and swaps.
% - ratesSet: A structure containing the rates for deposits, futures, and swaps.
%
% Outputs:
% - dates: A vector of dates for which discount factors are calculated.
% - discount: A vector of discount factors corresponding to the dates.

% Conventions:
deposYearfrac = 2;
IByearfrac = 3;
swapsYearfrac = 6;

% Depos
settlementDate = datesSet.settlement;
libor = mean(ratesSet.depos, 2); % Calculates Libor Rates as mean of Deposit rates
year_frac = yearfrac(settlementDate, datesSet.depos, deposYearfrac);

B = 1./(1 + year_frac.*libor); % Computes Discout Factors
B = B(1:4); % Eliminate last 2 since futures contracts are traded more
B_df = table(datesSet.depos(1:4), B, zeroRates(settlementDate, datesSet.depos(1:4), B), 'VariableNames', {'date', 'discount', 'zero'}); % Create a dataframe for the discount factors

% Futures
settlementDate = datesSet.settlement;
libor_fwd = mean(ratesSet.futures, 2);
year_frac_fwd = yearfrac(datesSet.futures(:,1), datesSet.futures(:,2), deposYearfrac);

B_fwd = 1./(1 + year_frac_fwd.*libor_fwd); % Computes Forward discount factors

r_interp = interp1(B_df.date(3:4), B_df.zero(3:4), datesSet.futures(1,1)); % Interpolate to get the 15/03/2023 discount factor
B_interp = exp(-r_interp*yearfrac(settlementDate, datesSet.futures(1,1), IByearfrac)); % Goes back to 15/03/2023 discount factor
B = B_interp * B_fwd(1); % B at 15/06/2023

dates = [B_df.date(1:4); datesSet.futures(1,2)]; % Extend the list of dates with futures expiry dates
rates = [B_df.zero(1:4); zeroRates(settlementDate, datesSet.futures(1,2), B)]; % Compute zero rates for the newly added dates

r_interp = interp1(dates, rates, datesSet.futures(2,1), 'previous', 'extrap'); % Extrapolate the zero rate for the 21/06/2023
B_ = exp(-r_interp*yearfrac(settlementDate, datesSet.futures(2,1), IByearfrac)); % B at 21/06/2023

B(2) = B_ * B_fwd(2); % B at 21/09/2023

settDates = datesSet.futures(1:end-2,1);
expDates = datesSet.futures(1:end-2,2);
zeroR = zeroRates(settlementDate, expDates(1:2), B');
for i = 3:length(B_fwd)-2
    diff = expDates(i-1) - settDates(i);
    if diff > 0 % interpolate
        r_interp = interp1(expDates(i-2:i-1), zeroR(i-2:i-1), settDates(i), 'linear');
        B(i) = exp(-r_interp*yearfrac(settlementDate, settDates(i), IByearfrac)) * B_fwd(i);
        zeroR(i) = zeroRates(settlementDate, expDates(i), B(i));
    end
    if diff < 0 % extrapolate
        disp("Extrapolation")
        r_interp = interp1(expDates(i-2:i-1), zeroR(i-2:i-1), settDates(i), 'previous', 'extrap');
        B(i) = exp(-r_interp*yearfrac(settlementDate, settDates(i), IByearfrac)) * B_fwd(i);
        zeroR(i) = zeroRates(settlementDate, expDates(i), B(i));
    end
    if diff == 0
        B(i) = B(i-1) * B_fwd(i); % fix for values 1 day
        zeroR(i) = zeroRates(settlementDate, expDates(i), B(i));
    end
end

B_df = tAdd(B_df, {expDates, B', zeroRates(settlementDate, expDates, B')});

% Swaps

knownDates = datesSet.swaps;
expiryDates = ExpiryDates(year(knownDates(1))-1, year(knownDates(end)), month(knownDates(1)-3), day(knownDates(1)-3)); % Sets all Expiry dates (all year missing in the markets [+2 days])

unknownDates = setdiff(datenum(expiryDates), knownDates); % gets element differing in the 2 arrays

swapRates = mean(ratesSet.swaps, 2);
interpolatedSwaps = spline(knownDates, swapRates, unknownDates);

swapKnown = table(knownDates, swapRates, 'VariableNames', {'date', 'swap_rate'});
swapUnknown = table(unknownDates, interpolatedSwaps, 'VariableNames', {'date', 'swap_rate'});
swapCombined = sortrows([swapKnown; swapUnknown], 'date');

N = length(swapCombined.date) + 1;
B = zeros(N, 1);

zeroR = interp1(B_df.date, B_df.zero, swapCombined.date(1), 'linear');
B(1) = exp(-zeroR*yearfrac(settlementDate, swapCombined.date(1), IByearfrac));

B(2) = (1 - swapCombined.swap_rate(2) * (yearfrac(settlementDate, swapCombined.date(1), swapsYearfrac)*B(1))) ...
    / (1 + yearfrac(swapCombined.date(1), swapCombined.date(2), swapsYearfrac) * swapCombined.swap_rate(2));

accrualPeriods = yearfrac([settlementDate; swapCombined.date(1:end-1)], swapCombined.date(1:end), swapsYearfrac);

for i = 3:N-1
    deltaT = yearfrac(swapCombined.date(i-1), swapCombined.date(i), swapsYearfrac);  % scalar
    B(i) = (1 - swapCombined.swap_rate(i) * sum(accrualPeriods(1:i-1) .* B(1:i-1))) / (1 + deltaT * swapCombined.swap_rate(i));
end

B_df_new = tAdd(B_df, {swapCombined.date, B(1:end-1), zeroRates(settlementDate, swapCombined.date, B(1:end-1))});
dates = B_df_new.date;
discount = B_df_new.discount;

end


