#### Entrainment-Detrainment kernels
include(joinpath("..","helper_funcs", "diagnose_environment.jl"))

function entr_detr(
    m::AtmosModel{FT},
    entr::EntrainmentDetrainment,
    state::Vars,
    aux::Vars,
    t::Real,
    i::Int,
) where {FT}

    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    gm_a = aux
    en_a = aux.turbconv.environment
    up_a = aux.turbconv.updraft

    N_upd = n_updrafts(m.turbconv)
    ρinv = 1 / gm.ρ
    up_area = up[i].ρa / gm.ρ

    sqrt_ϵ = sqrt(eps(FT))
    # precompute vars
    a_en = environment_area(state, aux, N_upd)
    w_en = environment_w(state, aux, N_upd)
    w_up = up[i].ρaw / up[i].ρa
    sqrt_tke = sqrt(max(en.ρatke,0) * ρinv / a_en)
    Δw = max(w_up - w_en, sqrt_ϵ)
    Δb = up_a[i].buoyancy - en_a.buoyancy

    D_ε, D_δ, M_δ, M_ε =
        nondimensional_exchange_functions(m, entr, state, aux, t, i)

    Λ_1 = abs(Δb/Δw)
    Λ_2 = entr.c_λ * abs(Δb / (max(en.ρatke,0) + sqrt_ϵ))
    Λ = SVector(Λ_1, Λ_2)
    lower_bound = FT(0.1) # need to be moved ?
    upper_bound = FT(0.0005)
    # λ = lamb_smooth_minimum(Λ, lower_bound, upper_bound)
    λ = abs(Δb/Δw)

    # compute entrainment/detrainmnet components
    # ε_trb = 2 * up_area * entr.c_t * sqrt_tke / max( (w_up * up_area * up_a[i].updraft_top),FT(1e-4))
    ε_trb = 2 * up_area * entr.c_t * sqrt_tke / max( (w_up * up_area * FT(500)),sqrt_ϵ)
    ε_dyn = λ / max(w_up, sqrt_ϵ) * (D_ε + M_ε)
    δ_dyn = λ / max(w_up, sqrt_ϵ) * (D_δ + M_δ)

    ε_lim = ε_limiter(up_area, sqrt_ϵ)
    δ_lim = δ_limiter(up_area, sqrt_ϵ)

    ε_dyn = min(max(ε_dyn,FT(0)), FT(0.01))*ε_lim
    δ_dyn = min(max(δ_dyn,FT(0)), FT(0.01))*δ_lim
    ε_trb = min(max(ε_trb,FT(0)), FT(0.01))

    return ε_dyn ,δ_dyn, ε_trb
end;

ε_limiter(a_up::FT, ϵ::FT) where {FT} = 1+10*(1-1/(1+exp(-FT(0.1)*a_up/ϵ)))
δ_limiter(a_up::FT, ϵ::FT) where {FT} = 1+10*(1-1/(1+exp(-FT(0.1)*(1-a_up)/ϵ)))

# ε_limiter(a_up::FT, ϵ::FT) where {FT} = 1
# δ_limiter(a_up::FT, ϵ::FT) where {FT} = 1

