# # TODO: This method should be generalizes and moved to the back-end

function compute_updraft_top!(
    dg::DGModel,
    m::AtmosModel{FT},
    turbconv::EDMF{FT},
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
) where {FT}
    N_up = n_updrafts(m.turbconv)
    vs_a_i = vars_state(m, Auxiliary(), UInt16)
    vim_a = Vars{vs_a_i}(1:varsize(vs_a_i))
    vs_c_i = vars_state(m, Prognostic(), UInt16)
    vim_c = Vars{vs_c_i}(1:varsize(vs_c_i))
    vs_a = vars_state(m, Auxiliary(), FT)
    vs_c = vars_state(m, Prognostic(), FT)

    grid = dg.grid
    aux = dg.state_auxiliary
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    z_all = get_z(dg.grid)
    z_min = min(z_all...)
    for i_up in 1:N_up
        aux[:, vim_a.turbconv.updraft[i_up].updraft_top, :] .= z_min
    end

    for eh in 1:nhorzelem
        for j in 1:Nq
            for i in 1:Nq
                for ev in 1:nvertelem
                    e = ev + (eh - 1) * nvertelem
                    for k in 1:Nqk
                        ijk = i + Nq * ((j - 1) + Nq * (k - 1))

                        # --------------- kernel
                        vars_a = Vars{vs_a}(aux[ijk, :, e])
                        vars_c = Vars{vs_c}(Q[ijk, :, e])
                        ρinv = 1 / vars_c.ρ
                        z = altitude(m, vars_a)
                        for i_up in 1:N_up
                            up_i_ρa = vars_c.turbconv.updraft[i_up].ρa
                            updraft_top = aux[
                                ijk,
                                vim_a.turbconv.updraft[i_up].updraft_top,
                                e,
                            ]
                            if up_i_ρa * ρinv > 0
                                aux[
                                    ijk,
                                    vim_a.turbconv.updraft[i_up].updraft_top,
                                    e,
                                ] = max(updraft_top, z)
                            end
                        end
                        # ---------------

                    end
                end
            end
        end
    end
    for i_up in 1:N_up
        aux[:, vim_a.turbconv.updraft[i_up].updraft_top, :] .= max(
            aux[end, vim_a.turbconv.updraft[i_up].updraft_top, end],
            FT(500),
        )
    end

end
