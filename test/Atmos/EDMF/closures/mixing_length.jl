#### Mixing length model kernels
include(joinpath("..","helper_funcs", "diagnose_environment.jl"))

function mixing_length(
    m::AtmosModel{FT},
    ml::MixingLengthModel,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    δ::AbstractArray{FT},
    εt::AbstractArray{FT},
) where {FT}

    # need to code / use the functions: obukhov_length, ustar, ϕ_m
    m.turbconv.surface
    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    gm_d = diffusive
    gm_a = aux
    en_d = diffusive.turbconv.environment
    N_upd = n_updrafts(m.turbconv)

    z = altitude(m, aux)
    _grav = FT(grav(m.param_set))
    ρinv = 1 / gm.ρ

    # precompute
    en_area  = environment_area(state, aux, N_upd)
    w_env    = environment_w(state, aux, N_upd)

    gm_d_∇u = diffusive.turbconv.∇u
    # TODO: check rank of `en_d.∇u`
    Shear = gm_d_∇u[1,3]^2 + gm_d_∇u[2,3]^2 + en_d.∇w[3]^2 # consider scalar product of two vectors
    tke = max(en.ρatke,0)*ρinv/en_area

    # bflux     = Nishizawa2018.compute_buoyancy_flux(m.param_set, ml.shf, ml.lhf, ml.T_b, q, ρinv)
    bflux = FT(1)
    θ_surf = m.turbconv.surface.T_surf
    # ustar = Nishizawa2018.compute_friction_velocity(m.param_set ,u_ave ,θ_suft ,flux ,Δz ,z_0 ,a ,Ψ_m_tol ,tol_abs ,iter_max)
    ustar = FT(0.28)
    # obukhov_length = Nishizawa2018.monin_obukhov_len(m.param_set, u, θ_surf, flux)
    obukhov_length = FT(-100)

    # buoyancy related functions
    ∂b∂z, Nˢ_eff = compute_buoyancy_gradients(m, state, diffusive, aux, t)
    Grad_Ri = gradient_Richardson_number(∂b∂z, Shear, FT(0.25)) # this parameter should be exposed in the model
    Pr_z = turbulent_Prandtl_number(FT(1), Grad_Ri)

    # compute L1
    if Nˢ_eff > eps(FT)
        L_1 = min(sqrt(ml.c_b * tke) / Nˢ_eff, 1e6)
    else
        L_1 = 1.0e6
    end

    # compute L2 - law of the wall  - YAIR define tke_surf
    # tke_surf = FT(3.75)*ustar*ustar
    θ_liq_cv_surf, q_tot_cv_surf, θ_liq_q_tot_cv_surf, tke_surf =
            env_surface_covariances(m.turbconv.surface, m.turbconv, m, gm, gm_a)
    if obukhov_length < FT(0)
        L_2 =
            (
                ml.κ * z / (
                    sqrt(tke_surf) / ustar / ustar
                ) * ml.c_m
            ) * min((FT(1) - FT(100) * z / obukhov_length)^FT(0.2), 1 / ml.κ)
    else
        L_2 =
            ml.κ * z / (
                sqrt(tke_surf) / ustar / ustar
            ) * ml.c_k
    end

    # compute L3 - entrainment detrainment sources
    # Production/destruction terms
    a = ml.c_m * (Shear - ∂b∂z/Pr_z) * sqrt(tke)
    # Dissipation term
    b = FT(0)
    for i in 1:N_upd
        a_up = up[i].ρa * ρinv
        w_up = up[i].ρaw / up[i].ρa
        b +=
            a_up * w_up * δ[i] / en_area *
            ((w_up - w_env) * (w_up - w_env) / 2 - tke) -
            a_up * w_up * (w_up - w_env) * εt[i] * w_env / en_area
    end

    c_neg = ml.c_d * tke * sqrt(abs(tke))
    if abs(a) > eps(FT) && 4 * a * c_neg > -b^2
        l_entdet =
            max(-b / FT(2) / a + sqrt(b^2 + 4 * a * c_neg) / 2 / a, FT(0))
    elseif abs(a) < eps(FT) && abs(b) > eps(FT)
        l_entdet = c_neg / b
    else
        l_entdet = FT(0)
    end
    L_3 = l_entdet

    l_mix = lamb_smooth_minimum(SVector(L_1,L_2,L_3))
    return l_mix
end;
