"""
    Zero-moment bulk microphysics scheme that instantly removes
    moisture above certain threshold.
    This is equivalent to instanteneous conversion of cloud condensate
    into precipitation and precipitation fallout with infinite
    terminal velocity.

"""
module Microphysics_0M

using ClimateMachine.Thermodynamics

using CLIMAParameters
using CLIMAParameters.Atmos.Microphysics

const APS = AbstractParameterSet

export remove_precipitation

"""
    remove_precipitation(param_set::APS, q; q_vap_sat)

 - `param_set` - abstract parameter set
 - `q` - current PhasePartition
 - `q_vap_sat` - water vapor specific humidity at saturation

Returns the `q_tot` tendency due to the removal of precipitation.
The tendency is obtained assuming a relaxation with a constant timescale
to a state with precipitable water removed.
The threshold for when to remove `q_tot` is defined either by the
condensate specific humidity or supersaturation.
The thresholds and the relaxation timescale are defined in
CLIMAParameters.
"""
# TODO:
# - τ_precip(param_set), etc - move to ClimaParameters
# - have dt here instead of constant timescale?

function remove_precipitation(
    param_set::APS,
    q::PhasePartition{FT},
) where {FT <: Real}

    _τ_precip::FT = FT(1000)
    _qc_0::FT = FT(5e-3)

    return -max(0, (q.liq + q.ice - _qc_0)) / _τ_precip
end

function remove_precipitation(
    param_set::APS,
    q::PhasePartition{FT},
    q_vap_sat::FT,
) where {FT <: Real}

    _τ_precip::FT = FT(1000)
    _S_0::FT = FT(0.02)

    return -max(0, (q.liq + q.ice - _S_0 * q_vap_sat)) / _τ_precip
end

end #module Microphysics_0M.jl
