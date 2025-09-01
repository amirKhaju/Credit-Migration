function [dates, discounts]=bootstrap(datesSet, ratesSet)
%% BOOTSTRAP FUNCTION
% This function computes the discount factors using the bootstrap method.
% The process involves extracting market rates for deposits, futures, and swaps,
% then using them to calculate discount factors iteratively.
%
% INPUTS:
%   datesSet  - Structure containing key dates for deposits, futures, and swaps.
%   ratesSet  - Structure containing bid-ask rates for deposits, futures, and swaps.
%
% OUTPUTS:
%   dates     - Vector of dates corresponding to calculated discount factors.
%   discounts - Vector of discount factors computed using bootstrap.

%% initializing variables

dates=zeros(61,1);      % array to store dates for discount factors
discounts=zeros(61,1);  % array to store discount factors
dates(1)=datesSet.settlement;   % settlement date
discounts(1)=1;     % initial discount factor

%% calculate Mid Market Rates
% The mid rates are the average of bid and ask rates.

rate_mid_depos=(ratesSet.depos(:,1)+ratesSet.depos(:,2))/(2);   % depos mid rate
rate_mid_futures=(ratesSet.futures(:,1)+ratesSet.futures(:,2))/2;   % futures mid rate

%%
% setting first dates
dates(2:5)=datesSet.depos(1:4);
dates(6:12)=datesSet.futures(1:7,2);

%% calculate first discounts using depos
discounts(2:5)=1./(1+(yearfrac(dates(1),dates(2:5),2).*rate_mid_depos(1:4)));
ratezz(1)=zeroRates(dates(4),discounts(4))/100;
ratezz(2)=zeroRates(dates(5),discounts(5))/100;


%%
% use linear interpolation to estimate the zero rate
zerorateinterpl = interp1(datesSet.depos(3:4), ratezz ,datesSet.futures(1,1), 'linear');

% DF fot the future using ACT/365 convention
discfut = exp(-yearfrac(datesSet.settlement, datesSet.futures(1,1),3)*zerorateinterpl);

% forward discount using ACT/360 convention
B_ti_tii=1/(1+rate_mid_futures(1)*yearfrac(datesSet.futures(1,1),datesSet.futures(1,2),2));
discounts(6)=discfut*B_ti_tii;

%% calculate DF related to futures using ACT/360 convention

for i=2:7

    B_ti_tii=1/(1+rate_mid_futures(i)*yearfrac(datesSet.futures(i,1),datesSet.futures(i,2),2));

    % control dates
    if datesSet.futures(i,1)==datesSet.futures(i-1,2)
        discounts(i+5)=discounts(i+4)*B_ti_tii;

    elseif(datesSet.futures(i,1) > datesSet.futures(i-1,2))
        % If settlement date of the future is greater than the expiry of the
        % previous one use formula for calculating the discount factor

        zerorate = zeroRates(dates(i+4), discounts(i+4))/100;
        B_t0_ti = exp(-yearfrac(datesSet.settlement, datesSet.futures(i,1),3)*zerorate);    % ACT/365 convention
        discounts(i+5)=B_t0_ti*B_ti_tii;

    else
        % else the settlement date of the future is precedent to the expiry of the
        % previous one use linear interpolation for calculating the discount factor

        zerorate = zeroRates(dates(i+3:i+4), discounts(i+3:i+4))/100;
        zerorateinterpl = interp1([dates(i+3:i+4)], zerorate,[dates(i+3:i+4);datesSet.futures(i,1)], 'linear');
        B_t0_ti = exp(-yearfrac(datesSet.settlement, datesSet.futures(i,1),3)*zerorateinterpl(3));
        discounts(i+5)=B_t0_ti*B_ti_tii;
    end

end

%% Compute Swap Rates and Discount Factors for Swaps
dateswap1y = datetime(2024,2,2);
dateswap1y=datenum(dateswap1y);
zerorate = zeroRates(dates(8:9), discounts(8:9))/100;
rateswaptemp=interp1(dates(8:9),zerorate,dateswap1y,"linear");

% Extract swap rates and dates
[datesswap, rateswaps]=swaperate(datesSet, ratesSet);

% Set swap payment dates
dates(13:13+length(datesswap)-1)=datesswap;

%% calculate DF related to swaps using 30/360 convention

b(1)=exp(-yearfrac(datesSet.settlement,dateswap1y,3)*rateswaptemp);
dateutili=[dateswap1y,datesswap];
sum=yearfrac(datesSet.settlement,dateutili(1),6)*b(1);

for i=1:49
    b(i+1)=(1-rateswaps(i)*sum)/(1+yearfrac(dateutili(i),dateutili(i+1),6)*rateswaps(i));
    sum=sum+yearfrac(dateutili(i),dateutili(i+1),6)*b(i+1);
end

discounts(13:length(datesswap)+13-1)=b(2:end);

end

