using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Filters
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.Diagnostics
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VTK

using CLIMA.Atmos: vars_state, vars_aux

using LinearAlgebra
using Random
using StaticArrays
using Logging
using Printf
using Dates

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray,)
else
  const ArrayTypes = (Array,)
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

"""
  Initial Condition for DYCOMS_RF01 LES
@article{doi:10.1175/MWR2930.1,
author = {Stevens, Bjorn and Moeng, Chin-Hoh and Ackerman,
          Andrew S. and Bretherton, Christopher S. and Chlond,
          Andreas and de Roode, Stephan and Edwards, James and Golaz,
          Jean-Christophe and Jiang, Hongli and Khairoutdinov,
          Marat and Kirkpatrick, Michael P. and Lewellen, David C. and Lock, Adrian and
          Maeller, Frank and Stevens, David E. and Whelan, Eoin and Zhu, Ping},
title = {Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus},
journal = {Monthly Weather Review},
volume = {133},
number = {6},
pages = {1443-1462},
year = {2005},
doi = {10.1175/MWR2930.1},
URL = {https://doi.org/10.1175/MWR2930.1},
eprint = {https://doi.org/10.1175/MWR2930.1}
}
"""
function Initialise_DYCOMS!(state::Vars, aux::Vars, (x,y,z), t)
  FT            = eltype(state)
  xvert::FT     = z
  Rd::FT        = R_d
  Rv::FT        = R_v
  Rm::FT        = Rd
  ϵdv::FT       = Rv/Rd
  cpd::FT       = cp_d

  # These constants are those used by Stevens et al. (2005)
  qref::FT      = FT(9.0e-3)
  q_tot_sfc::FT = qref
  q_pt_sfc      = PhasePartition(q_tot_sfc)
  Rm_sfc::FT    = 461.5 #gas_constant_air(q_pt_sfc) # 461.5
  T_sfc::FT     = 290.4
  P_sfc::FT     = MSLP
  ρ_sfc::FT     = P_sfc / Rm_sfc / T_sfc
  # Specify moisture profiles
  q_liq::FT      = 0
  q_ice::FT      = 0
  q_c::FT        = 0
  zb::FT         = 600         # initial cloud bottom
  zi::FT         = 840         # initial cloud top
  ziplus::FT     = 875
  dz_cloud       = zi - zb
  θ_liq::FT      = 289
  if xvert <= zi
    θ_liq = FT(289.0)
    q_tot = qref
  else
    θ_liq = FT(297.0) + (xvert - zi)^(FT(1/3))
    q_tot = FT(1.5e-3)
  end
  q_c = q_liq + q_ice
  #Rm  = Rd*(FT(1) + (ϵdv - FT(1))*q_tot - ϵdv*q_c)

  # --------------------------------------------------
  # perturb initial state to break the symmetry and
  # trigger turbulent convection
  # --------------------------------------------------
  #randnum1   = rand(seed, FT) / 100
  #randnum2   = rand(seed, FT) / 1000
  #randnum1   = rand(Uniform(-0.02,0.02), 1, 1)
  #randnum2   = rand(Uniform(-0.000015,0.000015), 1, 1)
  #if xvert <= 25.0
  #  θ_liq += randnum1 * θ_liq
  #  q_tot += randnum2 * q_tot
  #end
  # --------------------------------------------------
  # END perturb initial state
  # --------------------------------------------------

    # Calculate PhasePartition object for vertical domain extent
    q_pt  = PhasePartition(q_tot, q_liq, q_ice)
    Rm    = gas_constant_air(q_pt)

    # Pressure
    H     = Rm_sfc * T_sfc / grav;
    p     = P_sfc * exp(-xvert/H);
    # Density, Temperature
    # TODO: temporary fix
    TS    = LiquidIcePotTempSHumEquil_given_pressure(θ_liq, q_tot, p)
    ρ     = air_density(TS)
    T     = air_temperature(TS)
    q_pt  = PhasePartition_equil(T, ρ, q_tot)

  # Assign State Variables
  u1, u2 = FT(6), FT(7)
  v1, v2 = FT(-4.25), FT(-5.5)
  w = FT(0)
  if (xvert <= zi)
      u, v = u1, v1
  elseif (xvert >= ziplus)
      u, v = u2, v2
  else
      m = (ziplus - zi)/(u2 - u1)
      u = (xvert - zi)/m + u1

      m = (ziplus - zi)/(v2 - v1)
      v = (xvert - zi)/m + v1
  end
  e_kin       = FT(1/2) * (u^2 + v^2 + w^2)
  e_pot       = grav * xvert
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(ρ*u, ρ*v, ρ*w)
  state.ρe    = E
  state.moisture.ρq_tot = ρ * q_tot

end

function run(mpicomm, ArrayType, dim, topl, N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, xmax, ymax, zmax, zsponge, out_dir)
  # Grid setup (topl contains brickrange information)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  # Problem constants
  # Radiation model
  κ             = FT(85)
  α_z           = FT(1)
  z_i           = FT(840)
  D_subsidence  = FT(3.75e-6)
  ρ_i           = FT(1.13)
  F_0           = FT(70)
  F_1           = FT(22)
  # Geostrophic forcing
  f_coriolis    = FT(7.62e-5)
  u_geostrophic = FT(7.0)
  v_geostrophic = FT(-5.5)
  w_ref         = FT(0)
  u_relaxation  = SVector(u_geostrophic, v_geostrophic, w_ref)
  #Sponge:
  c_sponge = 1

  # Model definition
  model = AtmosModel(FlatOrientation(),
                     NoReferenceState(),
                     SmagorinskyLilly{FT}(C_smag),
                     EquilMoist(),
                     StevensRadiation{FT}(κ, α_z, z_i, ρ_i, D_subsidence, F_0, F_1),
                     (Gravity(),
                      RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation),
                      GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)),
                     DYCOMS_BC{FT}(C_drag, LHF, SHF),
                     Initialise_DYCOMS!)
  # Balancelaw description
  dg = DGModel(model,
               grid,
               Rusanov(),
               CentralNumericalFluxDiffusive(),
               CentralGradPenalty())
  #Q = init_ode_state(dg, FT(0); device=CPU())
  Q = init_ode_state(dg, FT(0))
  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
  # Calculating initial condition norm
 #= eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0
 =#
  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      #= energy = norm(Q) =#
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     """, ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"))
    end
  end


  # Setup VTK output callbacks
  out_interval = 5000
  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(out_interval) do (init=false)
    fprefix = @sprintf("dycoms_%dD_mpirank%04d_step%04d", dim,
                       MPI.Comm_rank(mpicomm), step[1])
    outprefix = joinpath(out_dir, fprefix)
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, dg, flattenednames(vars_state(model,FT)),
             dg.auxstate, flattenednames(vars_aux(model,FT)))

    step[1] += 1
    nothing
  end

  # Get statistics during run
  diagnostics_time_str = string(now())
  cbdiagnostics = GenericCallbacks.EveryXSimulationSteps(out_interval) do (init=false)
    sim_time_str = string(ODESolvers.gettime(lsrk))
    gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                       xmax, ymax, out_dir)
  end

  # Filter (Exponential via Callback)
  filterorder = 14
  filter_interval_exonential = 1
  cbexpfilter = GenericCallbacks.EveryXSimulationSteps(filter_interval_exonential) do
    # Applies exponential filter to all prognostic variables
    Filters.apply!(Q, 1:size(Q, 2), grid,ExponentialFilter(grid, 0, filterorder))
    nothing
  end

  # Filter (TMAR via Callback)
  filter_interval_TMAR = 1
  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(filter_interval_TMAR) do
    # Applies positivity cutoff to ρq_tot only
    Filters.apply!(Q, 6, disc.grid, TMARFilter())
    nothing
  end

    
  #solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk, cbdiagnostics))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk, cbdiagnostics))

  # Get statistics at the end of the run
  sim_time_str = string(ODESolvers.gettime(lsrk))
  gather_diagnostics(mpicomm, dg, Q, diagnostics_time_str, sim_time_str,
                     xmax, ymax, out_dir)

  # Print some end of the simulation information
 #= engf = norm(Q)
  Qe = init_ode_state(dg, FT(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  engf/eng0
=#
end

using Test
let
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD

  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  out_dir = get(ENV, "OUT_DIR", "output")
  mkpath(out_dir)

  @static if haspkg("CUDAnative")
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

 # @testset "$(@__FILE__)" for ArrayType in ArrayTypes
  for ArrayType in ArrayTypes
    # Problem type
    FT = Float64
    # DG polynomial order
    N = 4
    # SGS Filter constants
    C_smag = FT(0.21)
    LHF    = FT(115)
    SHF    = FT(15)
    C_drag = FT(0.0011)
    # User defined domain parameters
    Δx, Δy, Δz = 50, 50, 20
    #xmin, xmax = 0, 3200
    #ymin, ymax = 0, 3200
    xmin, xmax = 0, 1000
    ymin, ymax = 0, 1000
    zmin, zmax = 0, 1500

    grid_resolution = [Δx, Δy, Δz]
    domain_size     = [xmin, xmax, ymin, ymax, zmin, zmax]
    dim = length(grid_resolution)

    brickrange = (grid1d(xmin, xmax, elemsize=FT(grid_resolution[1])*N),
                  grid1d(ymin, ymax, elemsize=FT(grid_resolution[2])*N),
                  grid1d(zmin, zmax, elemsize=FT(grid_resolution[end])*N))
    zmax = brickrange[dim][end]

    zsponge = FT(1000.0)
    topl = StackedBrickTopology(mpicomm, brickrange,
                                periodicity = (true, true, false),
                                boundary=((0,0),(0,0),(1,2)))
    dt = 0.01
    timeend = 14400
    @info (ArrayType, dt, FT, dim)
    result = run(mpicomm, ArrayType, dim, topl,
                 N, timeend, FT, dt, C_smag, LHF, SHF, C_drag, xmax, ymax, zmax, zsponge,
                 out_dir)

  end
  @show LH_v0
  @show R_d
  @show MSLP
  @show cp_d
end

#include(joinpath("..","..","..","src","Diagnostics","graph_diagnostic.jl"))
nothing
