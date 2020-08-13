#### EDMF model

#### Entrainment-Detrainment model

Base.@kwdef struct EntrainmentDetrainment{FT<:AbstractFloat}
    "Entrainmnet TKE scale"
    c_λ::FT = 0.3
    "Entrainment factor"
    c_ε::FT = 0.13
    "Detrainment factor"
    c_δ::FT = 0.52
    "Trubulent Entrainment factor"
    c_t::FT = 0.1
    "Detrainment RH power"
    β::FT = 2
    "Logistic function scale ‵[1/s]‵"
    μ_0::FT = 0.0004
    "Updraft mixing fraction"
    χ::FT = 0.25
end

Base.@kwdef struct SurfaceModel{FT<:AbstractFloat}
    "Surface temperature ‵[k]‵"
    surface_T::FT = 300.4
    "Surface liquid water potential temperature ‵[k]‵"
    surface_θ_liq::FT = 299.1
    "Surface specific humidity ‵[kg/kg]‵"
    surface_q_tot::FT = 22.45e-3
    "surface sensible heat flux ‵[w/m^2]‵"
    surface_shf::FT = 9.5
    "surface lantent heat flux ‵[w/m^2]‵"
    surface_lhf::FT = 147.2
    "top sensible heat flux ‵[w/m^2]‵"
    top_shf::FT = 0.0
    "top lantent heat flux ‵[w/m^2]‵"
    top_lhf::FT = 0.0
    "Sufcae area"
    a_surf::FT = 0.1
    "Sufcae tempearture"
    T_surf::FT = 300.0
    "Ratio of rms turbulent velocity to friction velocity"
    κ_star::FT = 1.94
    "fixed ustar" # YAIR - need to change this
    ustar::FT = 0.28

end

Base.@kwdef struct PressureModel{FT<:AbstractFloat}
    "Pressure drag"
    α_d::FT = 10.0
    "Pressure advection"
    α_a::FT = 0.1
    "Pressure buoyancy"
    α_b::FT = 0.12
end

Base.@kwdef struct MixingLengthModel{FT<:AbstractFloat}
    "dissipation coefficient"
    c_d::FT = 0.22
    "Eddy Viscosity"
    c_m::FT = 0.14
    "Static Stability coefficient"
    c_b::FT = 0.63
    "Empirical stability function coefficient"
    a1::FT = -100
    "Empirical stability function coefficient"
    a2::FT = -0.2
    "Von karmen constant"
    κ::FT = 0.4
end

abstract type AbstractStatisticalModel end
struct SubdomainMean <: AbstractStatisticalModel end
struct GaussQuad <: AbstractStatisticalModel end
struct LogNormalQuad <: AbstractStatisticalModel end

Base.@kwdef struct MicrophysicsModel{FT<:AbstractFloat, SM}
    "enviromental cloud fraction"
    cf_initial::FT
    "Subdomain statistical model"
    statistical_model::SM
end

function MicrophysicsModel(FT;
        cf_initial = FT(0.0),
        statistical_model = SubdomainMean(),
    )
    args =(cf_initial,statistical_model)
    return MicrophysicsModel{FT,
        typeof(statistical_model)}(args...)
end

Base.@kwdef struct Environment{FT<:AbstractFloat, N_quad} <: BalanceLaw end

Base.@kwdef struct Updraft{FT<:AbstractFloat} <: BalanceLaw end

Base.@kwdef struct EDMF{FT<:AbstractFloat, N, UP, EN, ED, P, S, MP, ML} <: TurbulenceConvectionModel
    "Updrafts"
    updraft::UP
    "Environment"
    environment::EN
    "Entrainment-Detrainment model"
    entr_detr::ED
    "Pressure model"
    pressure::P
    "Surface model"
    surface::S
    "Microphysics model"
    micro_phys::MP
    "Mixing length model"
    mix_len::ML
end

function EDMF(FT, N, N_quad;
    updraft = ntuple(i -> Updraft{FT}(), N),
    environment = Environment{FT,N_quad}(),
    entr_detr = EntrainmentDetrainment{FT}(),
    pressure = PressureModel{FT}(),
    surface = SurfaceModel{FT}(),
    micro_phys = MicrophysicsModel(FT),
    mix_len = MixingLengthModel{FT}(),
    )
    args = (updraft,
    environment,
    entr_detr,
    pressure,
    surface,
    micro_phys,
    mix_len)
    return EDMF{FT, N, typeof.(args)...}(args...)
end


import ClimateMachine.TurbulenceConvection:
    turbconv_sources

n_updrafts(m::EDMF{FT, N}) where {FT, N} = N
n_quad_points(m::Environment{FT, N_quad}) where {FT, N_quad} = N_quad
turbconv_sources(m::EDMF) = (turbconv_source!,)

struct EDMFBCs <: TurbConvBC end
