using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Dierckx
using DelimitedFiles
using Logging, Printf, Dates
using CLIMA.VTK
using CLIMA.Mesh.Filters

using CLIMA.Atmos
import CLIMA.Atmos: MoistureModel, temperature, pressure, soundspeed, update_aux!
import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive! 
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!
# function pressure(m::MMSDryModel, state::Vars, aux::Vars)
#   T = eltype(state)
#   γ = T(7)/T(5)
#   ρinv = 1 / state.ρ
#   return (γ-1)*(state.ρe - ρinv/2 * sum(abs2, state.ρu))

# end

# function soundspeed(m::MMSDryModel, state::Vars, aux::Vars)
#   T = eltype(state)
#   γ = T(7)/T(5)
#   ρinv = 1 / state.ρ
#   p = pressure(m, state, aux)
#   sqrt(ρinv * γ * p)
# end

# TODO: Get this from geometry
const (xmin, xmax) = (-30000,30000)
const (ymin, ymax) = (0,5000)
const (zmin, zmax) = (0, 24000)
function init_state!(bl, state::Vars, aux::Vars, (x1,x2,x3), args...)
  spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit = args[2]
  FT         = eltype(state)

  x = x1
  y = x2
  z = x3
  xvert          = z
  datat          = FT(spl_tinit(xvert))
  dataq          = FT(spl_qinit(xvert))
  datau          = FT(spl_uinit(xvert))
  datav          = FT(spl_vinit(xvert))
  datap          = FT(spl_pinit(xvert))
  dataq          = dataq / 1000

  if xvert >= 14000
      dataq = 0.0
  end

  θ_c =     3.0
  rx  = 10000.0
  ry  =  2000.0
  rz  =  1500.0
  xc  = 0.0#0.5*(xmax + xmin)
  yc  = 2500.0
  zc  = 2000.0

  cylinder_flg = 0.0
  r   = sqrt( (x - xc)^2/rx^2 + cylinder_flg*(y - yc)^2/ry^2 + (z - zc)^2/rz^2)
  Δθ  = 0.0
  if r <= 1.0
      Δθ = θ_c * (cospi(0.5*r))^2
  end
  
  θ_liq = datat + Δθ
  q_tot = dataq
  p     = datap
  T     = air_temperature_from_liquid_ice_pottemp_given_pressure(θ_liq, p, PhasePartition(q_tot)) 
  ρ     = air_density(T, p)
  # energy definitions
  u, v, w     = datau, datav, zero(FT) #geostrophic. TO BE BUILT PROPERLY if Coriolis is considered
  ρu          = ρ * u
  ρv          = ρ * v
  ρw          = ρ * w
  e_kin       = (u^2 + v^2 + w^2) / 2
  e_pot       = grav * xvert
  ρe_tot      = ρ * total_energy(e_kin, e_pot, T, PhasePartition(q_tot))
  ρq_tot      = ρ * q_tot
  state.ρ = ρ
  state.ρu = SVector(ρu, ρv, ρw)
  state.ρe = ρe_tot
  state.moisture.ρq_tot = ρq_tot
  state.moisture.ρq_liq = FT(0)
  state.moisture.ρq_ice = FT(0)
  state.precipitation.ρq_rain = FT(0)
  nothing
end

function read_sounding()
    #read in the original squal sounding
    fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_gabersek.dat"))
    #fsounding  = open(joinpath(@__DIR__, "../soundings/sounding_gabersek_3deg_warmer.dat"))
    sounding = readdlm(fsounding)
    close(fsounding)
    (nzmax, ncols) = size(sounding)
    if nzmax == 0
        error("SOUNDING ERROR: The Sounding file is empty!")
    end
    return (sounding, nzmax, ncols)
end

function spline_int()

  # ----------------------------------------------------
  # GET DATA FROM INTERPOLATED ARRAY ONTO VECTORS
  # This driver accepts data in 6 column format
  # ----------------------------------------------------
  (sounding, _, ncols) = read_sounding()

  # WARNING: Not all sounding data is formatted/scaled
  # the same. Care required in assigning array values
  # height theta qv    u     v     pressure
  zinit, tinit, qinit, uinit, vinit, pinit  =
      sounding[:, 1], sounding[:, 2], sounding[:, 3], sounding[:, 4], sounding[:, 5], sounding[:, 6]
  #------------------------------------------------------
  # GET SPLINE FUNCTION
  #------------------------------------------------------
  spl_tinit    = Spline1D(zinit, tinit; k=1)
  spl_qinit    = Spline1D(zinit, qinit; k=1)
  spl_uinit    = Spline1D(zinit, uinit; k=1)
  spl_vinit    = Spline1D(zinit, vinit; k=1)
  spl_pinit    = Spline1D(zinit, pinit; k=1)
  return spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit
end


using CLIMA.Atmos: vars_state, vars_aux


function config_squall_line(FT, N, resolution, xmin, xmax, ymin, ymax, zmax)
rayleigh_sponge = RayleighSponge{FT}(zmax, 16500, 0.5, SVector{3,FT}(0,0,0), 2,2)
    model = AtmosModel{FT}(AtmosLESConfiguration;
                           turbulence=ConstantViscosityWithDivergence{FT}(100),
			       precipitation = Rain(),
			       ref_state = NoReferenceState(),
			       moisture = NonEquilMoist(),
                               source = (Gravity(),rayleigh_sponge,CloudSource(),PrecipitationSource()),
                    boundarycondition = Squall_lineBC(),
                           init_state = init_state!)


    config = CLIMA.Atmos_LES_Configuration("squall_line", N, resolution, xmax, ymax, zmax,
                                     init_state!,
				     xmin = xmin,
				     solver_type=CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
                                     model = model)

    return config
end
function main()
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δx = FT(250)
    Δy = FT(1000)
    Δz = FT(200)
    resolution = (Δx, Δy, Δz)

    t0 = FT(0)
    timeend = FT(9000)
    spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit = spline_int()
    driver_config = config_squall_line(FT, N, resolution, xmin, xmax, ymin, ymax, zmax)
    solver_config = CLIMA.setup_solver(t0, timeend, driver_config, (spl_tinit, spl_qinit, spl_uinit, spl_vinit, spl_pinit), forcecpu=true, Courant_number=1.1)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
        Filters.apply!(solver_config.Q, (6,7,8,9), solver_config.dg.grid, TMARFilter())
        nothing
    end
    result = CLIMA.invoke!(solver_config;
                          user_callbacks=(cbtmarfilter,),
                          check_euclidean_distance=true)
end

main()



#nothing