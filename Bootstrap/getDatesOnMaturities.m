function maturity_dates = getDatesOnMaturities(settlement_date, maturities)
    end_date = datenum(datetime(settlement_date, 'ConvertFrom', 'datenum') + years(maturities(end)+1));
    all_dates = getDates(settlement_date, end_date, 'quarterly', 'forward');
%    maturity_dates = [];

%    for i = 1:length(maturities)
%        date = datetime(settlement_date, 'ConvertFrom', 'datenum') + years(maturities(i));
%        for j = 1:length(all_dates)
%            if year(all_dates(j)) == year(date)
%                maturity_dates = [maturity_dates, all_dates(j)];
%            end
%        end
%    end
end