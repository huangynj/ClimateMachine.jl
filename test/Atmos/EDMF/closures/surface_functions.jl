#### Surface model kernels

## --- revert to use compute_buoyancy_flux in SurfaceFluxes.jl ---|

# function compute_blux(
#     ss::AtmosModel{FT, N},
#     m::SurfaceModel,
#     source::Vars,
#     state::Vars,
#     ) where {FT, N}

#     ts = PhaseEquil(param_set ,state.e_int, state.ρ, state.q_tot)
#     ϵ_v::FT       = 1 / molmass_ratio(param_set)
#     _T0::FT       = T_0(param_set)
#     _e_int_i0::FT = e_int_i0(param_set)
#     _grav::FT     = grav(param_set)
#     _cv_m::FT    = cv_m(ts)
#   return  _grav*( (m.e_int_surface_flux, -m.q_tot_surface_flux*_e_int_i0 )/(_cv_m*_T0 + state.e_int - state.q_tot*_e_int_i0 )
#                 + ( (ϵ_v-1)*m.q_tot_surface_flux)/(1+(ϵ_v-1)*state.q_tot)) # this equation should verified in the design docs
# end;
# function compute_MO_len(κ::FT, ustar::FT, bflux::FT) where {FT<:Real, PS}
#   return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar / bflux / κ
# end;

## --- revert to use compute_buoyancy_flux in SurfaceFluxes.jl ---|

using Statistics

function env_surface_covariances(
    m::SurfaceModel,
    turbconv::EDMF{FT},
    atmos::AtmosModel{FT},
    state::Vars,
    aux::Vars,
) where {FT}
    turbconv = atmos.turbconv
    N = n_updrafts(turbconv)
    # yair - I would like to call the surface functions from src/Atmos/Model/SurfaceFluxes.jl
    # bflux = Nishizawa2018.compute_buoyancy_flux(ss.param_set, m.shf, m.lhf, T_b, q, α_0) # missing def of m.shf, m.lhf, T_b, q, α_0
    # oblength = Nishizawa2018.monin_obukhov_len(ss.param_set, u, θ, bflux) # missing def of u, θ,
    # ustar = Nishizawa2018.ustar(..)
    # for now I am using fixed values
    # override ------------------
    # gm_p = air_pressure(thermo_state(atmos, state, aux))

    # e_int = internal_energy(atmos, state, aux)
    # ts = PhaseEquil(atmos.param_set, e_int, state.ρ, state.moisture.ρq_tot / state.ρ)

    ts = thermo_state(atmos, state, aux)
    gm_p = air_pressure(ts)
    θ_liq = liquid_ice_pottemp(ts)
    q = PhasePartition(ts)
    _cp_m = cp_m(atmos.param_set, q)
    lv = latent_heat_vapor(ts)
    Π = exner(ts)

    θ_liq_surface_flux = m.surface_shf/Π/_cp_m
    q_tot_surface_flux = m.surface_lhf/lv
    oblength = FT(-100)
    ustar = FT(0.28)
    zLL = FT(20) # how to get the z first interior ?
    if oblength < 0
      θ_liq_cv       = 4 * (θ_liq_surface_flux*θ_liq_surface_flux)/(ustar*ustar) * (1 - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
      q_tot_cv       = 4 * (q_tot_surface_flux*q_tot_surface_flux)/(ustar*ustar) * (1 - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
      θ_liq_q_tot_cv = 4 * (θ_liq_surface_flux*q_tot_surface_flux)/(ustar*ustar) * (1 - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
      tke            = ustar * ustar * (FT(3.75) + cbrt(zLL/oblength * zLL/oblength))
    else
      θ_liq_cv       = 4 * (θ_liq_surface_flux * θ_liq_surface_flux)/(ustar*ustar)
      q_tot_cv       = 4 * (q_tot_surface_flux * q_tot_surface_flux)/(ustar*ustar)
      θ_liq_q_tot_cv = 4 * (θ_liq_surface_flux * q_tot_surface_flux)/(ustar*ustar)
      tke            = FT(3.75) * ustar * ustar
    end
    return θ_liq_cv, q_tot_cv, θ_liq_q_tot_cv, tke
end;

function compute_updraft_surface_BC(
    m::SurfaceModel,
    turbconv::EDMF{FT},
    atmos::AtmosModel{FT},
    state::Vars,
    aux::Vars,
) where {FT}
    turbconv = atmos.turbconv
    N = n_updrafts(turbconv)
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    ρinv = 1 / gm.ρ

    θ_liq_cv, q_tot_cv, θ_liq_q_tot_cv, tke =
        env_surface_covariances(m, turbconv, atmos, state, aux)
    upd_a_surf = MArray{Tuple{N}, FT}(zeros(FT, N))
    upd_θ_liq_surf = MArray{Tuple{N}, FT}(zeros(FT, N))
    upd_q_tot_surf = MArray{Tuple{N}, FT}(zeros(FT, N))
    for i in 1:N
        surface_scalar_coeff = percentile_bounds_mean_norm(1 - m.a_surf + (i-1) * FT(m.a_surf/N),
                                                           1 - m.a_surf + i     * FT(m.a_surf/N), 1000)
        upd_a_surf[i] = FT(m.a_surf/N)
        e_int = internal_energy(atmos, state, aux)
        ts = PhaseEquil(atmos.param_set, e_int, state.ρ, state.moisture.ρq_tot / state.ρ)
        gm_θ_liq = liquid_ice_pottemp(ts)
        upd_θ_liq_surf[i] = (gm_θ_liq                + surface_scalar_coeff*sqrt(θ_liq_cv))
        upd_q_tot_surf[i] = (gm.moisture.ρq_tot*ρinv + surface_scalar_coeff*sqrt(q_tot_cv))
    end
    return upd_a_surf, upd_θ_liq_surf, upd_q_tot_surf
end;

function percentile_bounds_mean_norm(
    low_percentile::FT,
    high_percentile::FT,
    n_samples::IT,
) where {FT <: Real, IT}
    x = rand(Normal(), n_samples)
    xp_low = quantile(Normal(), low_percentile)
    xp_high = quantile(Normal(), high_percentile)
    filter!(y -> xp_low < y < xp_high, x)
    return Statistics.mean(x)
end
