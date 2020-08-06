"""
    SoilHeatParameterizations

Functions for volumetric heat capacity, internal energy as function of temperature,
saturated thermal conductivity, thermal conductivty, the Kersten number, and relative
saturation are included.
"""

module SoilHeatParameterizations

using DocStringExtensions

export volumetric_heat_capacity,
    internal_energy,
    saturated_thermal_conductivity,
    thermal_conductivity,
    relative_saturation,
    kersten_number,
    internal_energy_liquid_water,
    temperature_from_I


"""
    function temperature_from_I(
        T_ref::FT,
        I::FT,
        θ_ice::FT,
        ρ_ice::FT,
        LH_f_0::FT,
        cs::FT
    ) where {FT}

Computes the temperature given I and `θ_ice`.
"""
function temperature_from_I(
    T_ref::FT,
    I::FT,
    θ_ice::FT,
    ρ_ice::FT,
    LH_f0::FT,
    cs::FT
) where {FT}
    T = FT(T_ref + (I + θ_ice*ρ_ice*LH_f0)/cs)
    return T
end


"""
    volumetric_heat_capacity(
        ϴ_l::FT,
        ϴ_i::FT,
        c_ds::FT,
        cp_l::FT,
        cp_i::FT
    ) where {FT}
Compute the expression for volumetric heat capacity.
"""
function volumetric_heat_capacity(
    ϴ_l::FT,
    ϴ_i::FT,
    c_ds::FT,
    cp_l::FT,
    cp_i::FT
) where {FT}

    c_s = FT(c_ds + ϴ_l *cp_l + ϴ_i * cp_i)
    return c_s
end

"""
    internal_energy(
        ϴ_i::FT,
        c_s::FT,
        T::FT,
        T_ref::FT,
        ρ_i::FT,
        LH_f_0::FT
    ) where {FT}
Compute the expression for internal energy.
"""
function internal_energy(
    ϴ_i::FT,
    c_s::FT,
    T::FT,
    T_ref::FT,
    ρ_i::FT,
    LH_f_0::FT
) where {FT}
    I = FT(c_s * (T - T_ref) - ϴ_i * ρ_i * LH_f_0)
    return I
end

"""
    saturated_thermal_conductivity(
        ϴ_l::FT,
        ϴ_i::FT,
        porosity::FT,
        κ_sat_unfrozen::FT,
        κ_sat_frozen::FT
    ) where {FT}
Compute the expression for saturated thermal conductivity of soil matrix.
"""
function saturated_thermal_conductivity(
    ϴ_l::FT,
    ϴ_i::FT,
    κ_sat_unfrozen::FT,
    κ_sat_frozen::FT
) where {FT}
    #TBD: can we get rid of this branch? if not: create test for it.
    θ_w = ϴ_l + ϴ_i
    if θ_w <= eps(FT)
        κ_sat = FT(0.0)
    else
        κ_sat = FT(κ_sat_unfrozen^(ϴ_l / ϴ_w) * κ_sat_frozen^(ϴ_i / ϴ_w))
    end
    
    return κ_sat
end

"""
    relative_saturation(
            ϑ_l::FT,
            ϴ_i::FT,
            porosity::FT
    ) where {FT}
Compute the expression for relative saturation.
"""
function relative_saturation(
    θ_l::FT,
    ϴ_i::FT,
    porosity::FT
) where {FT}

    S_r=FT((θ_l + ϴ_i) / porosity)
    return S_r
end

"""
    kersten_number(
        ϴ_i::FT,
        S_r::FT,
        a::FT,
        b::FT,
        ν_om::FT,
        ν_sand::FT,
        ν_gravel::FT
    ) where {FT}
Compute the expression for the Kersten number.
"""
function kersten_number(
    ϴ_i::FT,
    S_r::FT,
    a::FT,
    b::FT,
    ν_om::FT,
    ν_sand::FT,
    ν_gravel::FT
) where {FT}

    if ϴ_i <= eps(FT)
        K_e = FT(S_r^((1 + ν_om - a * ν_sand - ν_gravel) / 2) * ((1 + exp(-b * S_r))^(-3) - ((1 - S_r) / 2)^3)^(1 - ν_om))
    else
        K_e = FT(S_r^(1 + ν_om))
    end
    return K_e
end

"""
    thermal_conductivity(
        κ_dry::FT,
        K_e::FT,
        κ_sat::FT
    ) where {FT}
Compute the expression for thermal conductivity of soil matrix.
"""
function thermal_conductivity(
    κ_dry::FT,
    K_e::FT,
    κ_sat::FT
) where {FT}

    κ = FT(K_e * κ_sat + (1 - K_e) * κ_dry)
    return κ
end

"""
    internal_energy_liquid_water(
        cp_l::FT,
        T::FT,
        T_ref::FT,
        ρ_l::FT
    ) where {FT}
Compute the expression for the internal energy of liquid water.
"""
function internal_energy_liquid_water(
    cp_l::FT,
    T::FT,
    T_ref::FT,
    ρ_l::FT
) where {FT}

    I_l = FT(ρ_l * cp_l * (T - T_ref))
    return I_l
end

end # Module