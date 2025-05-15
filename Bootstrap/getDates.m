function dates = getDates(startDate, endDate, jump, mode)
    % jumb can be yearly or monthly
    % mode can be forward or backward
    if isnumeric(startDate)
        startDate = datetime(startDate, 'ConvertFrom', 'datenum');
    else
        startDate = datetime(startDate, 'InputFormat', 'dd/MM/yyyy');
    end

    if isnumeric(endDate)
        endDate = datetime(endDate, 'ConvertFrom', 'datenum');
    else
        endDate = datetime(endDate, 'InputFormat', 'dd/MM/yyyy');
    end
    startYear = year(startDate);
    endYear = year(endDate);
    lenYear = endYear - startYear;
    dates = NaT(1, 0); % not a time
    dates.Format = startDate.Format;
    switch jump
        case 'yearly'
            for i=1:lenYear+1
                dates(i) = startDate + calyears(i-1); % jump of 1 year
                dates(i) = checkWorkingDay(dates(i), mode);
            end
        case 'quarterly'
            for i = 1:(lenYear * 4)+1
                dates(i) = startDate + calmonths((i-1) * 3); % jump of 3 months
                dates(i) = checkWorkingDay(dates(i), mode);
            end
    end
    dates = dates';
end
    
    
function newDate = checkWorkingDay(date, mode)
    found = false;
    while ~found
        if ~isNonWorkingDay(date)
            newDate = date;
            found = true;
        else
            if strcmpi(mode, 'forward')
                date = date + caldays(1); % forward 1 day
            elseif strcmpi(mode, 'backward')
                date = date - caldays(1); % backward 1 day
            else
                error('Use mode: "forward" or "backward".');
            end
        end
    end
end


function flag = isNonWorkingDay(dt)
    yr = year(dt);
    dn = datenum(dt);
    holidayDates = [datenum(yr,1,1),...   % New Year's day
                    datenum(yr,2,22),...  % President day
                    datenum(yr,7,4),...   % Indipendence day
                    datenum(yr,12,25),... % Christmas day
                    datenum(yr,12,26)];   % Saint Stefano
    easterDates = [ datenum(2008, 3, 23), datenum(2009, 4, 12), datenum(2010, 4, 4),  datenum(2011, 4, 24), ...
                    datenum(2012, 4, 8),  datenum(2013, 3, 31), datenum(2014, 4, 20), datenum(2015, 4, 5),  ...
                    datenum(2016, 3, 27), datenum(2017, 4, 16), datenum(2018, 4, 1),  datenum(2019, 4, 21), ...
                    datenum(2020, 4, 12), datenum(2021, 4, 4),  datenum(2022, 4, 17), datenum(2023, 4, 9),  ...
                    datenum(2024, 3, 31), datenum(2025, 4, 20), datenum(2026, 4, 5),  datenum(2027, 3, 28), ...
                    datenum(2028, 4, 16), datenum(2029, 4, 1),  datenum(2030, 4, 21), datenum(2031, 4, 13), ...
                    datenum(2032, 3, 28), datenum(2033, 4, 17), datenum(2034, 4, 9),  datenum(2035, 3, 25), ...
                    datenum(2036, 4, 13), datenum(2037, 4, 5),  datenum(2038, 4, 25), datenum(2039, 4, 10), ...
                    datenum(2040, 4, 1),  datenum(2041, 4, 21), datenum(2042, 4, 6),  datenum(2043, 3, 29), ...
                    datenum(2044, 4, 17), datenum(2045, 4, 9),  datenum(2046, 3, 25), datenum(2047, 4, 14), ...
                    datenum(2048, 4, 5),  datenum(2049, 4, 18), datenum(2050, 4, 10), datenum(2051, 3, 26), ...
                    datenum(2052, 4, 21), datenum(2053, 4, 6),  datenum(2054, 3, 29), datenum(2055, 4, 18), ...
                    datenum(2056, 4, 10), datenum(2057, 4, 22), datenum(2058, 4, 14), datenum(2059, 3, 30), ...
                    datenum(2060, 4, 18), datenum(2061, 4, 10), datenum(2062, 3, 26), datenum(2063, 4, 15), ...
                    datenum(2064, 4, 6),  datenum(2065, 3, 22), datenum(2066, 4, 11), datenum(2067, 4, 3),  ...
                    datenum(2068, 4, 22), datenum(2069, 4, 7),  datenum(2070, 3, 30), datenum(2071, 4, 19), ...
                    datenum(2072, 4, 10), datenum(2073, 3, 26), datenum(2074, 4, 15) ]; % Easter days from 2008 to 2074
    venerdiSanto = false;
    if weekday(dn) == 6 && ismember(dn+2, easterDates)
        venerdiSanto = true;
    end
    flag = (weekday(dn) == 1 || weekday(dn) == 7 || ismember(dn, holidayDates) || venerdiSanto);
end
