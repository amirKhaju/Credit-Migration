function targetDiscounts = getDiscounts(dates, discounts, targetDates, settlementDate)
    r_interp = interp1(dates, zeroRates(settlementDate, dates, discounts), targetDates, 'linear');
    targetDiscounts = exp(-r_interp.*yearfrac(settlementDate, targetDates, 3));
    if isnan(targetDiscounts(1))
        targetDiscounts(1) = 1;
    end
end