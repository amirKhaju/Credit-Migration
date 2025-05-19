function dates = getDates(startDate, endDate, jump, mode)
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
        dates = startDate + calyears(0:lenYear);
    case 'quarterly'
        dates = startDate + calmonths(0:3:(lenYear * 4) * 3);
    case 'semi_annual'
        dates = startDate + calmonths(0:6:(lenYear * 2)*6);
end
dates = arrayfun(@(d) checkWorkingDay(d, mode), dates);

dates = dates';
end

function newDate = checkWorkingDay(date, mode)
    arguments
        date (1,1) datetime
        mode (1,:) char {mustBeMember(mode, {'forward', 'backward'})}
    end

    step = caldays(1);
    if strcmpi(mode, 'backward')
        step = -step;
    end

    while isNonWorkingDay(date)
        date = date + step;
    end

    newDate = date;
end


function flag = isNonWorkingDay(dt)
    persistent easterDates
    
    dn = datenum(dt);
    yr = year(dt);

    if isempty(easterDates)
        easterDates = datenum([ ...
            2008, 3, 23; 2009, 4, 12; 2010, 4, 4; 2011, 4, 24;
            2012, 4, 8;  2013, 3, 31; 2014, 4, 20; 2015, 4, 5;
            2016, 3, 27; 2017, 4, 16; 2018, 4, 1;  2019, 4, 21;
            2020, 4, 12; 2021, 4, 4;  2022, 4, 17; 2023, 4, 9;
            2024, 3, 31; 2025, 4, 20; 2026, 4, 5;  2027, 3, 28;
            2028, 4, 16; 2029, 4, 1;  2030, 4, 21; 2031, 4, 13;
            2032, 3, 28; 2033, 4, 17; 2034, 4, 9;  2035, 3, 25;
            2036, 4, 13; 2037, 4, 5;  2038, 4, 25; 2039, 4, 10;
            2040, 4, 1;  2041, 4, 21; 2042, 4, 6;  2043, 3, 29;
            2044, 4, 17; 2045, 4, 9;  2046, 3, 25; 2047, 4, 14;
            2048, 4, 5;  2049, 4, 18; 2050, 4, 10; 2051, 3, 26;
            2052, 4, 21; 2053, 4, 6;  2054, 3, 29; 2055, 4, 18;
            2056, 4, 10; 2057, 4, 22; 2058, 4, 14; 2059, 3, 30;
            2060, 4, 18; 2061, 4, 10; 2062, 3, 26; 2063, 4, 15;
            2064, 4, 6;  2065, 3, 22; 2066, 4, 11; 2067, 4, 3;
            2068, 4, 22; 2069, 4, 7;  2070, 3, 30; 2071, 4, 19;
            2072, 4, 10; 2073, 3, 26; 2074, 4, 15 ]);
    end

    holidayDates = datenum([ ...
        yr, 1, 1;   % New Year's Day
        yr, 2, 22;  % President's Day
        yr, 7, 4;   % Independence Day
        yr, 12, 25; % Christmas Day
        yr, 12, 26  % Saint Stefano
    ]);

    % Check for Friday before Easter (Good Friday)
    isGoodFriday = (weekday(dn) == 6) && ismember(dn + 2, easterDates);

    % Final check: weekend, fixed holidays, Easter, or Good Friday
    isWeekend = weekday(dn) == 1 || weekday(dn) == 7;
    isHoliday = ismember(dn, holidayDates);
    isEaster = ismember(dn, easterDates);

    flag = isWeekend || isHoliday || isEaster || isGoodFriday;
end
