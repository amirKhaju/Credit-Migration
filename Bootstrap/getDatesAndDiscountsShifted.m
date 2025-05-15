function [datesVector, discountsShifted] = getDatesAndDiscountsShifted(datesSet, datesVector, ratesVector, bpvShift, final_date_idx) 
    delta = bpvShift * 1e-4;
    discountsShifted = zeros(length(datesVector), length(ratesVector.depos)+length(ratesVector.futures)+length(ratesVector.swaps));
    for i = 1:length(ratesVector.depos)
        if i < final_date_idx
            break;
        end
        ratesVector.depos(i, :) = ratesVector.depos(i, :) + delta;
        [datesVector, temp] = bootstrap(datesSet, ratesVector);
        ratesVector.depos(i, :) = ratesVector.depos(i, :) - delta;
        discountsShifted(:, i) = temp;
    end
    for i = 1:length(ratesVector.futures)
        if i < final_date_idx + length(ratesVector.depos)
            break;
        end
        ratesVector.futures(i, :) = ratesVector.futures(i, :) + delta;
        [datesVector, temp] = bootstrap(datesSet, ratesVector);
        ratesVector.futures(i, :) = ratesVector.futures(i, :) - delta;
        discountsShifted(:, i + length(ratesVector.depos)) = temp;
    end
    for i = 1:length(ratesVector.swaps)
        if i < final_date_idx + length(ratesVector.depos) + length(ratesVector.futures)
            break;
        end
        ratesVector.swaps(i, :) = ratesVector.swaps(i, :) + delta;
        [datesVector, temp] = bootstrap(datesSet, ratesVector);
        ratesVector.swaps(i, :) = ratesVector.swaps(i, :) - delta;
        discountsShifted(:, i + length(ratesVector.futures) + length(ratesVector.depos)) = temp;
    end
end