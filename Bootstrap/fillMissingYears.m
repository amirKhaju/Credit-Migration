function [newVol, newYears] = fillMissingYears(volMatrix, maturities)
    % fillMissingYears Riempi anni mancanti con NaN
    %
    %   [newVol, newYears] = fillMissingYears(volMatrix, maturities)
    %
    % INPUT:
    %   volMatrix  : matrice (N×M) di volatilità, una riga per ogni maturity
    %   maturities : vettore (N×1 o 1×N) di anni interi corrispondenti alle righe
    %
    % OUTPUT:
    %   newVol   : matrice (max_year × M) con righe da 1 a max_year;
    %              anni senza dati originali sono NaN
    %   newYears : vettore riga [1 2 … max_year]
    
        % Assicuriamoci maturities come vettore colonna
        maturities = maturities(:);
    
        % Trovo anno massimo
        maxYear = max(maturities);
    
        % Nuovo vettore anni
        newYears = 1:maxYear;
    
        % Numero di colonne di volMatrix
        nCols = size(volMatrix, 2);
    
        % Preallocazione con NaN
        newVol = NaN(maxYear, nCols);
    
        % Copio i dati nelle righe corrispondenti
        for i = 1:length(maturities)
            yr = maturities(i);
            if yr >= 1 && yr <= maxYear
                newVol(yr, :) = volMatrix(i, :);
            end
        end
    end