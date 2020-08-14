# Test heat equation agrees with analytic solution to problem 55 on page 28 in https://ocw.mit.edu/courses/mathematics/18-303-linear-partial-differential-equations-fall-2006/lecture-notes/heateqni.pdf
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Test

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
import ClimateMachine.DGMethods: calculate_dt

function calculate_dt(dg, model::LandModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end
function diffusive_courant(
    m::LandModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    return Δt * m.soil.param_functions.κ_dry / (Δx * Δx)
end

@testset "Heat analytic unit test" begin
    ClimateMachine.init()
    FT = Float32

    function init_soil!(land, state, aux, coordinates, time)
        myFT = eltype(state)
        _ρ_l = myFT(ρ_cloud_liq(param_set))
        _ρ_i = myFT(ρ_cloud_ice(param_set))
        _cp_l = myFT(cp_l(param_set) * _ρ_l)
        _cp_i = myFT(cp_i(param_set) * _ρ_i)
        _ρ_i = myFT(ρ_cloud_ice(param_set))
        _T_ref = myFT(T_0(param_set))
        _LH_f0 = myFT(LH_f0(param_set))

        ϑ_l, θ_ice = get_water_content(land.soil.water, aux, state, time)
        θ_l =
            volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
        c_s = volumetric_heat_capacity(
            θ_l,
            θ_ice,
            land.soil.param_functions.c_ds,
            _cp_l,
            _cp_i,
        )

        state.soil.heat.I = myFT(internal_energy(
            θ_ice,
            c_s,
            land.soil.heat.initialT(aux),
            _T_ref,
            _ρ_i,
            _LH_f0,
        ))
    end

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.495,
        ν_gravel = 0.1,
        ν_om = 0.1,
        ν_sand = 0.1,
        c_ds = 1,
        κ_dry = 1,
        κ_sat_unfrozen = 0.57,
        κ_sat_frozen = 2.29,
        a = 0.24,
        b = 18.1,
    )

    zero_output = FT(0.0)
    surface_value = FT(0.0)
    heat_surface_state = (aux, t) -> surface_value

    tau = FT(1) # period (sec)
    A = FT(5) # ampltitude (T)
    ω = FT(2 * pi / tau)
    heat_bottom_state = (aux, t) -> A * cos(ω * t)

    initial_temp = FT(0.0)
    T_init = (aux) -> initial_temp

    soil_water_model =
        PrescribedWaterModel((aux, t) -> zero_output, (aux, t) -> zero_output)

    soil_heat_model = SoilHeatModel(
        FT;
        initialT = T_init,
        dirichlet_bc = Dirichlet(
            surface_state = heat_surface_state,
            bottom_state = heat_bottom_state,
        ),
        neumann_bc = Neumann(surface_flux = nothing, bottom_flux = nothing),
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil!,
    )

    N_poly = 5
    nelem_vert = 10

    # Specify the domain boundaries
    zmax = FT(1)
    zmin = FT(0)

    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    )

    t0 = FT(0)
    timeend = FT(5)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        Courant_number = FT(0.7),
        CFL_direction = VerticalDirection(),
    )
    mygrid = solver_config.dg.grid
    aux = solver_config.dg.state_auxiliary

    ClimateMachine.invoke!(solver_config)
    t = ODESolvers.gettime(solver_config.solver)

    z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
    z = Array(aux[:, z_ind, :][:])

    T_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
    T = Array(aux[:, T_ind, :][:])

    num =
        exp.(sqrt(ω / 2) * (1 + im) * (1 .- z)) .-
        exp.(-sqrt(ω / 2) * (1 + im) * (1 .- z))
    denom = exp(sqrt(ω / 2) * (1 + im)) - exp.(-sqrt(ω / 2) * (1 + im))
    analytic_soln = real(num .* A * exp(im * ω * timeend) / denom)
    MSE = mean((analytic_soln .- T) .^ 2.0)
    @test eltype(aux) == FT
    @test MSE < 1e-5
end
