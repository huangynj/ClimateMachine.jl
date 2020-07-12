using MPI
using OrderedCollections
using StaticArrays
using Statistics

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
# using ClimateMachine.Land.SoilHeatParameterizations
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

# This has the interpolation functions
include("./helperfunc.jl")

FT = Float64;

# When using ClimateMachine, we need to define a function that sets the initial
# state of our model run.

function init_soil!(land, state, aux, coordinates, time)
    FT = eltype(state)
    state.soil.water.ϑ = FT(land.soil.water.initialϑ(aux))
    state.soil.water.θ_ice = FT(land.soil.water.initialθ_ice(aux))
    state.soil.heat.I = FT(land.soil.heat.params.ρc * aux.soil.heat.T) # land.soil.heat.initialT(aux))
    #    state.ρu = SVector{3, FT}(0, 0, 0) might be a useful ref later for how to initialize vectors.
end;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

## Set initial condition

SoilParams = SoilParamSet(
        porosity = 0.495,
        Ksat = 0.0443 / (3600*100),
        S_s = 1e-3,
        ν_gravel = 0.2,
        ν_om = 0.2,
        ν_sand = 0.2,
        c_ds = 1e6,
        κ_dry = 1.5,
        κ_sat_unfrozen = 3.0,
        κ_sat_frozen = 3.5,
        ρc = 1.0,
        α = 0.01
        )
# Keep in mind that what is passed is aux⁻
# Fluxes are multiplied by ẑ (normal to the surface, -normal to the bottom,
# where normal points out of the domain.)
water_surface_state = (aux, t) -> FT(0.494)
water_bottom_flux = (aux, t) -> FT(aux.soil.water.κ*1.0)
ϑ_0 = (aux) -> FT(0.24)

heat_surface_state = (aux, t) -> FT(300)
heat_bottom_flux = (aux, t) -> FT(0)
T_0 = (aux) -> FT(280)

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = Haverkamp{FT}(),
    params = SoilParams,
    initialϑ = ϑ_0,
    dirichlet_bc = Dirichlet(
        surface_state = water_surface_state,
        bottom_state = nothing,
    ),
    neumann_bc = Neumann(
        surface_flux = nothing,
        bottom_flux = water_bottom_flux
    ),
)

soil_heat_model = SoilHeatModel(
    FT;
    params = SoilParams,
    initialT = T_0,
    dirichlet_bc = Dirichlet(
        surface_state = heat_surface_state,
        bottom_state = nothing
    ),
    neumann_bc = Neumann(
        surface_flux = nothing,
        bottom_flux = heat_bottom_flux
    ),
)

m_soil = SoilModel(soil_water_model, soil_heat_model)
sources = ()
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_conservative = init_soil!,
)

N_poly = 5;
nelem_vert = 10;

# Specify the domain boundaries
zmax = FT(0);
zmin = FT(-1)

driver_config = ClimateMachine.SingleStackConfiguration(
    "LandModel",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m;
    zmin = zmin,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

t0 = FT(0)
timeend = FT(60 * 60 * 24)

dt = FT(6)

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;

ClimateMachine.invoke!(solver_config);
t = ODESolvers.gettime(solver_config.solver)
state_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    Q,
    vars_state_conservative(m, FT),
)
aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    aux,
    vars_state_auxiliary(m, FT);
)
all_vars = OrderedDict(state_vars..., aux_vars...);
all_vars["t"] = [t]

# Compare with Bonan simulation data at 1 day.
bonan_moisture = reverse([
    0.493,
    0.492,
    0.489,
    0.487,
    0.484,
    0.480,
    0.475,
    0.470,
    0.463,
    0.455,
    0.446,
    0.434,
    0.419,
    0.402,
    0.383,
    0.361,
    0.339,
    0.317,
    0.297,
    0.280,
    0.267,
    0.257,
    0.251,
    0.247,
    0.244,
    0.242,
    0.241,
    0.241,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
    0.240,
])
bonan_z = reverse([
    -0.500,
    -1.500,
    -2.500,
    -3.500,
    -4.500,
    -5.500,
    -6.500,
    -7.500,
    -8.500,
    -9.500,
    -10.500,
    -11.500,
    -12.500,
    -13.500,
    -14.500,
    -15.500,
    -16.500,
    -17.500,
    -18.500,
    -19.500,
    -20.500,
    -21.500,
    -22.500,
    -23.500,
    -24.500,
    -25.500,
    -26.500,
    -27.500,
    -28.500,
    -29.500,
    -30.500,
    -31.500,
    -32.500,
    -33.500,
    -34.500,
    -35.500,
    -36.500,
    -37.500,
    -38.500,
    -39.500,
    -40.500,
    -41.500,
    -42.500,
    -43.500,
    -44.500,
    -45.500,
    -46.500,
    -47.500,
    -48.500,
    -49.500,
    -50.500,
    -51.500,
    -52.500,
    -53.500,
    -54.500,
    -55.500,
    -56.500,
    -57.500,
    -58.500,
    -59.500,
    -60.500,
    -61.500,
    -62.500,
    -63.500,
    -64.500,
    -65.500,
    -66.500,
    -67.500,
    -68.500,
    -69.500,
    -70.500,
    -71.500,
    -72.500,
    -73.500,
    -74.500,
    -75.500,
    -76.500,
    -77.500,
    -78.500,
    -79.500,
    -80.500,
    -81.500,
    -82.500,
    -83.500,
    -84.500,
    -85.500,
    -86.500,
    -87.500,
    -88.500,
    -89.500,
    -90.500,
    -91.500,
    -92.500,
    -93.500,
    -94.500,
    -95.500,
    -96.500,
    -97.500,
    -98.500,
    -99.500,
])
bonan_z = bonan_z ./ 100.0


# Create an interpolation from the Bonan data
bonan_moisture_continuous = TimeContinuousData(bonan_z, bonan_moisture)
bonan_at_clima_z = [bonan_moisture_continuous(i) for i in all_vars["z"]]
#this is not quite a true L2, because our z values are not equally spaced.
MSE = mean((bonan_at_clima_z .- all_vars["soil.water.ϑ"]) .^ 2.0)
@test MSE < 1e-5