#### Pressure model kernels
include(joinpath("..","helper_funcs", "diagnose_environment.jl"))

function perturbation_pressure(
    m::AtmosModel{FT},
    press::PressureModel,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    i::Int,
) where {FT}

    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    up_a = aux.turbconv.updraft
    up_d = diffusive.turbconv.updraft

    ρinv    = 1 / gm.ρ
    N_upd   = n_updrafts(m.turbconv)
    en_area = environment_area(state, aux, N_upd)
    w_env   = environment_w(state, aux, N_upd)
    w_up    = up[i].ρaw / up[i].ρa

    nh_press_buoy   = press.α_b * up_a[i].buoyancy
    nh_pressure_adv =-press.α_a * w_up * up_d[i].∇w[3]
    nh_pressure_drag= press.α_d * (w_up - w_env) * abs(w_up - w_env) / FT(500) #up_a[i].updraft_top

    dpdz = nh_press_buoy + nh_pressure_adv + nh_pressure_drag
    dpdz_tke_i = up[i].ρa * (w_up - w_env) * dpdz

    return dpdz, dpdz_tke_i
end;
