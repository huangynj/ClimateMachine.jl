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

FT = Float64;

function init_soil_water!(land, state, aux, coordinates, time)
    FT = eltype(state)
    state.soil.water.ϑ = FT(land.soil.water.initialϑ(aux))
    state.soil.water.θ_ice = FT(land.soil.water.initialθ_ice(aux))
end;


ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

struct HeatModel end

SoilParams = SoilParamSet(porosity = 0.75, Ksat = 1e-7, S_s = 1e-3)
bottom_flux = (aux, t) -> FT(-3.0*sin(pi*2.0*t/300.0)*aux.soil.water.κ)
surface_flux = nothing
surface_state = (aux, t) -> FT(0.2)
bottom_state = nothing
ϑ_0 = (aux) -> FT(0.2)
soil_water_model = SoilWaterModel(
    FT;
    params = SoilParams,
    initialϑ = ϑ_0,
    dirichlet_bc = Dirichlet(
        surface_state = surface_state,
        bottom_state = bottom_state
    ),
    neumann_bc = Neumann(
        surface_flux = surface_flux,
        bottom_flux = bottom_flux
    ),
)

m_soil = SoilModel(soil_water_model, HeatModel())
sources = ()
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_conservative = init_soil_water!,
)


N_poly = 5;
nelem_vert = 50;


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
timeend = FT(300)
dt = FT(0.05)

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
mygrid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;

const n_outputs = 30;

const every_x_simulation_time = ceil(Int, timeend / n_outputs);

all_data = Dict([k => Dict() for k in 0:n_outputs]...)

step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    solver_config.solver,
) do (init = false)
    t = ODESolvers.gettime(
        solver_config.solver
    )
    grads = SingleStackUtils.get_vars_from_nodal_stack(
        mygrid,
        solver_config.dg.state_gradient_flux,
        vars_state_gradient_flux(m, FT),
    )    
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
    all_vars = OrderedDict(state_vars..., aux_vars..., grads...)
    all_vars["t"]= [t]
    all_data[step[1]] = all_vars

    step[1] += 1
    nothing
end;

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

t = ODESolvers.gettime(solver_config.solver)
state_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    Q,
    vars_state_conservative(m, FT),
)
grads = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    solver_config.dg.state_gradient_flux,
    vars_state_gradient_flux(m, FT),
)
aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
    mygrid,
    aux,
    vars_state_auxiliary(m, FT);
)
all_vars = OrderedDict(state_vars..., aux_vars...,grads...);
all_vars["t"] = [t]
all_data[n_outputs] = all_vars


computed_bottom_∇h = [all_data[k]["soil.water.κ∇h[3]"][1] for k in 0:n_outputs]./[all_data[k]["soil.water.κ"][1] for k in 0:n_outputs]


t = [all_data[k]["t"][1] for k in 0:n_outputs]
prescribed_bottom_∇h  = t -> FT(-3.0*sin(pi*2.0*t/300.0))

MSE = mean((prescribed_bottom_∇h.(t) .- computed_bottom_∇h) .^ 2.0)
@test MSE < 1e-7