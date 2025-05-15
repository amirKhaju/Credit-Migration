function T = tAdd(T_old, data)
% tAdd - Adds new rows to an existing table and sorts the table by the first column.
%
% Syntax: T = tAdd(T_old, data)
%
% Inputs:
%    T_old - The original table to which new rows will be added.
%    data - A cell array containing the new data to be added. The length of data must match the number of columns in T_old.
%
% Outputs:
%    T - The updated table with the new rows added and sorted by the first column.
if length(data) ~= width(T_old)
    error('Il numero di vettori dati deve essere uguale al numero di colonne di T_old');
end
newRows = table(data{:}, 'VariableNames', T_old.Properties.VariableNames);
T = sortrows([T_old; newRows], T_old.Properties.VariableNames{1});
end