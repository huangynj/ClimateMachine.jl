#### EDMF model

#### Entrainment-Detrainment model

Base.@kwdef struct EntrainmentDetrainment{FT}
    "Fractional scales"
    Λ::MArray{Tuple{2}, FT} = MArray{Tuple{2}, FT}([0, 0])
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

Base.@kwdef struct SurfaceModel{FT}
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

Base.@kwdef struct PressureModel{FT}
    "Pressure drag"
    α_d::FT = 10.0
    "Pressure advection"
    α_a::FT = 0.1
    "Pressure buoyancy"
    α_b::FT = 0.12
end

Base.@kwdef struct MixingLengthModel{FT}
    "Mixing lengths"
    L::MArray{Tuple{3}, FT} = MArray{Tuple{3}, FT}([0, 0, 0])
    "Eddy Viscosity"
    c_m::FT = 0.14
    "Eddy Diffusivity"
    c_k::FT = 0.22
    "Static Stability coefficient"
    c_b::FT = 0.63
    "Empirical stability function coefficient"
    a1::FT = -100
    "Empirical stability function coefficient"
    a2::FT = -0.2
    "Von karmen constant"
    κ::FT = 0.4
end

Base.@kwdef struct MicrophysicsModel{FT}
    "dry stract"
    dry::MArray{Tuple{6}, FT} = MArray{Tuple{6}, FT}([0, 0, 0, 0, 0, 0])
    "cloudy stract"
    cloudy::MArray{Tuple{6}, FT} = MArray{Tuple{6}, FT}([0, 0, 0, 0, 0, 0])
    "enviromental cloud fraction"
    cf_initial::FT = 0.0 # need to define a function for cf
    "Subdomain statistical mmodel"
    statistical_model::String = "SubdomainMean"
    # now N_quad is replacing quadrature order and is passed from single_stack_test.jl

    # "quadrature order" # can we code it that if order is 1 than we get mean ?  do we need the gaussian option?
    # quadrature_order::FT = 3# yair needs to be a string: "mean", "gaussian quadrature", lognormal quadrature"
end

Base.@kwdef struct Environment{FT, N_quad} <: BalanceLaw end

Base.@kwdef struct Updraft{FT} <: BalanceLaw end

Base.@kwdef struct EDMF{FT, N, N_quad} <: TurbulenceConvectionModel
    "Updrafts"
    updraft::NTuple{N, Updraft{FT}} = ntuple(i -> Updraft{FT}(), N)
    "Environment"
    environment::Environment{FT, N_quad} = Environment{FT, N_quad}()
    "Entrainment-Detrainment model"
    entr_detr::EntrainmentDetrainment{FT} = EntrainmentDetrainment{FT}()
    "Pressure model"
    pressure::PressureModel{FT} = PressureModel{FT}()
    "Surface model"
    surface::SurfaceModel{FT} = SurfaceModel{FT}()
    "Surface model"
    micro_phys::MicrophysicsModel{FT} = MicrophysicsModel{FT}()
    "Mixing length model"
    mix_len::MixingLengthModel{FT} = MixingLengthModel{FT}()
end

n_updrafts(m::EDMF{FT, N, N_quad}) where {FT, N, N_quad} = N
turbconv_sources(m::EDMF) = (turbconv_source!,)

struct EDMFBCs <: TurbConvBC end