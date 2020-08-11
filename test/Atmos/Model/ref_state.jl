include("get_atmos_ref_states.jl")

@testset "Hydrostatic reference states" begin

    RH = 0.5
    # Fails on (80, 1)
    for (nelem_vert, N_poly) in [(40, 2), (20, 4)]
        z, all_data = get_atmos_ref_states(nelem_vert, N_poly, RH)
        phase_type = PhaseEquil
        T = all_data["ref_state.T"]
        p = all_data["ref_state.p"]
        ρ = all_data["ref_state.ρ"]
        q_tot = all_data["ref_state.ρq_tot"] ./ ρ
        q_pt = PhasePartition.(q_tot)

        # TODO: test that ρ and p are in discrete hydrostatic balance

        # Test state for thermodynamic consistency (with ideal gas law)
        @test all(
            T .≈
            air_temperature_from_ideal_gas_law.(Ref(param_set), p, ρ, q_pt),
        )

        # Test that relative humidity in reference state is approximately
        # input relative humidity
        RH_ref = relative_humidity.(Ref(param_set), T, p, Ref(phase_type), q_pt)
        @show max(abs.(RH .- RH_ref)...)
        @test all(isapprox.(RH, RH_ref, atol = 0.05))
    end

end
