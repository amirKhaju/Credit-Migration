function expiryDates = ExpiryDates(start_year, end_year, value_month, value_day, frequency)
% ExpiryDates calculates the expiry dates based on the given start year, end year, 
% value month, value day, and frequency.
%
% Syntax:
%   expiryDates = ExpiryDates(start_year, end_year, value_month, value_day, frequency)
%
% Inputs:
%   start_year  - The starting year for the calculation (integer).
%   end_year    - The ending year for the calculation (integer).
%   value_month - The month of the base date (integer).
%   value_day   - The day of the base date (integer).
%   frequency   - The frequency of the expiry dates ('year', 'quarterly', 'monthly').
%                 If not provided, defaults to 'year'.
%
% Outputs:
%   expiryDates - A datetime array containing the calculated expiry dates.

% Se frequency non viene passato, di default è 'year'
if nargin < 5
    frequency = 'year';
end

switch lower(frequency)
    case 'year'
        years = start_year:end_year;
        expiryDates = datetime(zeros(length(years),1), 'ConvertFrom', 'datenum');
        for idx = 1:length(years)
            y = years(idx);
            baseDate = datenum(y, value_month, value_day);
            expDay = baseDate + 2; % settlement date
            % If ExpDay is on Saturday, Sunday or Holiday (Closed Market), we move to the next business day
            while isNonWorkingDay(expDay)
                expDay = expDay + 1;
            end
            expiryDates(idx) = datetime(expDay, 'ConvertFrom', 'datenum');
        end

    case 'quarterly'
        expiryDatesSerial = [];
        i = 0;
        while true
            % Calculate the base month for the current quarter
            newMonth = value_month + 3 * i;
            yearOffset = floor((newMonth - 1) / 12);
            newMonth = mod(newMonth - 1, 12) + 1;
            quarterYear = start_year + yearOffset;
            if quarterYear > end_year
                break;
            end

            baseDate = datenum(quarterYear, newMonth, value_day);
            expDay = baseDate + 2;
            while isNonWorkingDay(expDay)
                expDay = expDay + 1;
            end
            expiryDatesSerial = [expiryDatesSerial; expDay];
            i = i + 1;
        end
        expiryDates = datetime(expiryDatesSerial, 'ConvertFrom', 'datenum');

    case 'monthly'
        expiryDatesSerial = [];
        i = 0;
        while true
            newMonth = value_month + i;
            yearOffset = floor((newMonth - 1) / 12);
            newMonth = mod(newMonth - 1, 12) + 1;
            currentYear = start_year + yearOffset;
            if currentYear > end_year
                break;
            end

            baseDate = datenum(currentYear, newMonth, value_day);
            expDay = baseDate + 2;
            while isNonWorkingDay(expDay)
                expDay = expDay + 1;
            end
            expiryDatesSerial = [expiryDatesSerial; expDay];
            i = i + 1;
        end
        expiryDates = datetime(expiryDatesSerial, 'ConvertFrom', 'datenum');

    otherwise
        error('Valore di frequency non riconosciuto. Usare "year", "quarterly" o "monthly".');
end

expiryDates.Format = 'dd/MM/yyyy';
end

function flag = isNonWorkingDay(dn)
% This funtion is for calculating the Non Working Days

    dt = datetime(dn, 'ConvertFrom', 'datenum');
    yr = year(dt);
    holidayDates = [datenum(yr,1,1), datenum(yr,2,22), datenum(yr,7,4), ...
                        datenum(yr,12,25), datenum(yr,12,26)];
    flag = (weekday(dn) == 1 || weekday(dn) == 7 || ismember(dn, holidayDates));
end