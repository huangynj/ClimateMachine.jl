using LambertW
# TODO: Add documentation
function lamb_smooth_minimum(
    l::AbstractArray{FT,N};
    lower_bound::FT = FT(0.1),
    frac_upper_bound::FT = FT(1.5),
) where {FT,N}

    leng = size(l)
    xmin = minimum(l)

    lambda0 = max(
        FT(xmin) * lower_bound /
        FT(real(LambertW.lambertw(FT(2) / MathConstants.e))),
        frac_upper_bound,
    )

    num = 0
    den = 0
    for i in 1:length(l)
        num += l[i] * exp(-(l[i] - xmin) / lambda0)
        den += exp(-(l[i] - xmin) / lambda0)
    end
    smin = num / den

    return smin
end
