# Convenience wrapper
save_subdomain_temperature!(m, state, aux) =
    save_subdomain_temperature!(m,m.moisture,state,aux)

using KernelAbstractions: @print

function save_subdomain_temperature!(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    N_up = n_updrafts(m.turbconv)
    ts_gm = thermo_state(m, state, aux)
    p = air_pressure(ts_gm)
    up = state.turbconv.updraft

    ρ = state.ρ
    ρinv = 1/state.ρ
    θ_liq_gm = liquid_ice_pottemp(ts_gm)
    a_en = 1 - sum([up[j].ρa for j in 1:N_up]) * ρinv
    θ_liq_en = (θ_liq_gm - sum([up[j].ρaθ_liq * ρinv for j in 1:N_up]))/a_en
    q_tot_gm = total_specific_humidity(ts_gm)
    q_tot_en = (q_tot_gm - sum([up[j].ρaq_tot * ρinv for j in 1:N_up]))/a_en
    for i in 1:N_up
        ρa_up = up[i].ρa
        ρaθ_liq_up = up[i].ρaθ_liq
        ρaq_tot_up = up[i].ρaq_tot
        θ_liq_up = ρaθ_liq_up / ρa_up
        q_tot_up = ρaq_tot_up / ρa_up
        try
            ts_up = LiquidIcePotTempSHumEquil_given_pressure(m.param_set, θ_liq_up, p, q_tot_up)
            aux.turbconv.updraft[i].T = air_temperature(ts_up)
        catch
            @print("************************************* sat adjust failed (updraft)\n")
            @show i
            @show altitude(m, aux)
            @show ts_gm
            @show ρa_up
            @show ρa_up * ρinv
            @show p,ρ
            @show liquid_ice_pottemp(ts_gm)
            @show total_specific_humidity(ts_gm)
            ts_up = LiquidIcePotTempSHumEquil_given_pressure(m.param_set, θ_liq_up, p, q_tot_up)
        end
    end
    try
        ts_en = LiquidIcePotTempSHumEquil_given_pressure(m.param_set, θ_liq_en, p, q_tot_en)
        aux.turbconv.environment.T = air_temperature(ts_en)
    catch
        @print("************************************* sat adjust failed (env)\n")
        for i in 1:N_up
            @print i
            @show θ_liq_en
            @show θ_liq_gm
            @show up[i].ρa
            @show up[i].ρaw
            @show up[i].ρaθ_liq
            @show up[i].ρaq_tot
        end
        @show altitude(m, aux)
        @show θ_liq_en
        @show q_tot_en
        @show ts_gm
        @show p,ρ
        @show liquid_ice_pottemp(ts_gm)
        @show total_specific_humidity(ts_gm)
        ts_en = LiquidIcePotTempSHumEquil_given_pressure(m.param_set, θ_liq_en, p, q_tot_en)
    end
    return nothing
end

# Convenience wrapper
thermo_state_up(m, state, aux, i_up) =
    thermo_state_up(m,m.moisture,state,aux,i_up)

function thermo_state_up(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    i_up::Int
    )
    FT = eltype(state)
    param_set = m.param_set
    up = state.turbconv.updraft

    ts_gm = thermo_state(m, state, aux)
    p = air_pressure(ts_gm)
    T = aux.turbconv.updraft[i_up].T
    q_tot = up[i_up].ρaq_tot / up[i_up].ρa
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot, PhaseEquil)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end

# Convenience wrapper
thermo_state_en(m, state, aux) =
    thermo_state_en(m, m.moisture, state, aux)

function thermo_state_en(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    )
    FT = eltype(state)
    param_set = m.param_set
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft

    ts_gm = thermo_state(m, state, aux)
    p = air_pressure(ts_gm)
    T = aux.turbconv.environment.T
    ρinv = 1/state.ρ
    ρaq_tot_en = total_specific_humidity(ts_gm) - sum([up[i].ρaq_tot for i in 1:N_up])*ρinv
    a_en = 1 - sum([up[i].ρa for i in 1:N_up])*ρinv
    q_tot = ρaq_tot_en * ρinv / a_en
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot, PhaseEquil)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end
