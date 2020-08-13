include(joinpath("..","helper_funcs", "diagnose_environment.jl"))

function nondimensional_exchange_functions(
    m::AtmosModel{FT},
    entr::EntrainmentDetrainment,
    state::Vars,
    aux::Vars,
    t::Real,
    i::Int,
) where {FT, N}

    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    gm_a = aux
    up_a = aux.turbconv.updraft
    en_a = aux.turbconv.environment

    # precompute vars=
    N_upd = n_updrafts(m.turbconv)
    ρinv = 1 / gm.ρ
    up_area = up[i].ρa * ρinv
    w_up = up[i].ρaw / up[i].ρa
    a_en = environment_area(state, aux, N_upd)
    w_en = environment_w(state, aux, N_upd)
    # thermodynamic variables
    ts = thermo_state(m, state, aux)
    ts_up = thermo_state_up(m, state, aux, i)
    RH_up = relative_humidity(ts_up)
    ts_en = thermo_state_en(m, state, aux)
    RH_en = relative_humidity(ts_en)

    Δw = max(w_up - w_en, 1e-4)
    Δb = up_a[i].buoyancy - en_a.buoyancy

    if saturated(ts_up) || saturated(ts_en)
        c_δ = entr.c_δ
    else
        c_δ = FT(0)
    end
    # compute dry and moist aux functions
    μ_ij = (entr.χ - up_area / (up_area + a_en))*Δb/Δw
    D_ε  = entr.c_ε /(1 + exp(-μ_ij/entr.μ_0))
    M_ε  = c_δ * (max((RH_en^entr.β - RH_up^entr.β), 0))^(1/entr.β)
    D_δ  = entr.c_ε /(1 + exp( μ_ij/entr.μ_0))
    M_δ  = c_δ * (max((RH_up^entr.β - RH_en^entr.β), 0))^(1/entr.β)
    return D_ε, D_δ, M_δ, M_ε
end;

saturated(ts::Thermodynamics.ThermodynamicState) = (relative_humidity(ts) ≈ 1)
