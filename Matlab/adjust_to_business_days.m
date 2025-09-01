function date = adjust_to_business_days(date)
% This function takes in input a vector of dates and returns the same
% vector with non business days adjusted to business days
%%

% Adjust dates to business days
date(~isbusday(date)) = busdate(date(~isbusday(date)));


end