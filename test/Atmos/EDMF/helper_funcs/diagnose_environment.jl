function environment_area(
    state::Vars,
    aux::Vars,
    N_up::Int,
) where {FT}
    return 1 - sum(ntuple(i->state.turbconv.updraft[i].ρa, N_up))/ state.ρ
end

function environment_w(
    state::Vars,
    aux::Vars,
    N_up::Int,
) where {FT}
    ρinv = 1/state.ρ
    a_en = environment_area(state ,aux ,N_up)
    return (state.ρu[3] - sum(ntuple(i->state.turbconv.updraft[i].ρaw, N_up)))/a_en*ρinv
end

function grid_mean_b(
    state::Vars,
    aux::Vars,
    N_up::Int,
) where {FT}
    ρinv = 1/state.ρ
    a_en = environment_area(state ,aux ,N_up)
    up = state.turbconv.updraft
    en_a = aux.turbconv.environment
    up_a = aux.turbconv.updraft
    return a_en * en_a.buoyancy + sum(ntuple(i->up_a[i].buoyancy*up[i].ρa*ρinv, N_up))
end