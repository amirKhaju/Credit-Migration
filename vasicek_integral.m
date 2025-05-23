function p = vasicek_integral(x, y, rho)
%% cancella funzione
    integrand = @(ys) normcdf((x - sqrt(rho)*ys) / sqrt(1 - rho)) .* ...
                      normcdf((y - sqrt(rho)*ys) / sqrt(1 - rho)) .* ...
                      normpdf(ys);
    p = integral(integrand, -10, 10); % -Inf,Inf sostituito da ±10 per stabilità
end
