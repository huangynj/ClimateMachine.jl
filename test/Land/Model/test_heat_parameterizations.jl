using MPI
using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land.SoilHeatParameterizations

@testset "Land heat parameterizations" begin
    FT = Float64;
    # Density of liquid water (kg/m``^3``)
    _ρ_l = FT(ρ_cloud_liq(param_set))
    # Density of ice water (kg/m``^3``)
    _ρ_i = FT(ρ_cloud_ice(param_set))
    # Volum. isoboric heat capacity liquid water (J/m3/K)
    _cp_l = FT(cp_l(param_set) * _ρ_l)
    # Volumetric isoboric heat capacity ice (J/m3/K)
    _cp_i = FT(cp_i(param_set) * _ρ_i)
    # Reference temperature (K)
    _T_ref = FT(T_0(param_set))
    # Latent heat of fusion at ``T_0`` (J/kg)
    _LH_f0 = FT(LH_f0(param_set))

    @test round(volumetric_heat_capacity(
        0.25,
        0.05,
        1e6,
        _cp_l,
        _cp_i
    ),sigdigits=5) == FT(2.1415e+06)

    @test round(internal_energy(
        0.05,
        2.1415e+06,
        300.0,
        _T_ref,
        _ρ_i,
        _LH_f0
    ),sigdigits=5) == FT(4.2187e+07)

    @test round(saturated_thermal_conductivity(
        0.25,
        0.05,
        0.57,
        2.29
    ), sigdigits=4) == FT(0.7187)

    @test round(relative_saturation(
        0.25,
        0.05,
        0.4
    ), sigdigits=2) == FT(0.75)

    # Test branching in kersten_number

    # ice fraction = 0
    @test round(kersten_number(
        0.0,
        0.75,
        0.24,
        18.1,
        0.1,
        0.1,
        0.1
    ), sigdigits=4) == FT(0.8675)

    # ice fraction ~= 0
    @test round(kersten_number(
        0.05,
        0.75,
        0.24,
        18.1,
        0.1,
        0.1,
        0.1
    ), sigdigits=4) == FT(0.7287)

    @test round(thermal_conductivity(
        1.5,
        0.7287,
        0.7187
    ), sigdigits=3) == FT(0.931)

    @test round(internal_energy_liquid_water(
        _cp_l,
        300.0,
        _T_ref,
        _ρ_l
    ), sigdigits=4) == FT(1.122e+11)

end
