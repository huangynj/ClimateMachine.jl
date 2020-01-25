using .NumericalFluxes: GradNumericalPenalty,
                        NumericalFluxNonDiffusive,
                        NumericalFluxDiffusive,
                        numerical_flux_nondiffusive!,
                        numerical_boundary_flux_nondiffusive!,
                        numerical_flux_diffusive!,
                        numerical_boundary_flux_diffusive!,
                        gradient_boundary_penalty!,
                        gradient_penalty!,
                        divergence_penalty!,
                        divergence_boundary_penalty!,
                        divergence_penalty!,
                        numerical_boundary_flux_hyperdiffusive!,
                        numerical_flux_hyperdiffusive!

using ..Mesh.Geometry
using ..Mesh.Grids: EveryDirection, VerticalDirection, HorizontalDirection

using Requires
@init @require CUDAnative = "be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
  using .CUDAnative
end

# {{{ FIXME: remove this after we've figure out how to pass through to kernel
const _ξ1x1, _ξ2x1, _ξ3x1 = Grids._ξ1x1, Grids._ξ2x1, Grids._ξ3x1
const _ξ1x2, _ξ2x2, _ξ3x2 = Grids._ξ1x2, Grids._ξ2x2, Grids._ξ3x2
const _ξ1x3, _ξ2x3, _ξ3x3 = Grids._ξ1x3, Grids._ξ2x3, Grids._ξ3x3
const _M, _MI = Grids._M, Grids._MI
const _x1, _x2, _x3 = Grids._x1, Grids._x2, Grids._x3
const _JcV = Grids._JcV

const _n1, _n2, _n3 = Grids._n1, Grids._n2, Grids._n3
const _sM, _vMI = Grids._sM, Grids._vMI
# }}}

"""
    volumerhs!(bl::BalanceLaw, Val(polyorder), rhs, Q, Qvisc, auxstate,
               vgeo, t, D, elems)

Computational kernel: Evaluate the volume integrals on right-hand side of a
`DGBalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function volumerhs!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, ::direction,
                    rhs, Q, Qvisc, Qhypervisc_grad, auxstate, vgeo, t,
                    ω, D, elems, increment) where {dim, polyorder, direction}
  N = polyorder
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)
  
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nhyperviscstate = num_hyperdiffusive(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_F = @shmem FT (3, Nq, Nq, Nqk, nstate)
  s_ω = @shmem FT (Nq, )
  s_half_D = @shmem FT (Nq, Nq)
  l_rhs = @scratch FT (nstate, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nstate}, FT}(undef))
  l_Q = @scratch FT (nstate, Nq, Nq, Nqk) 3
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_Qhypervisc = MArray{Tuple{nhyperviscstate}, FT}(undef)
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3
  l_F = MArray{Tuple{3, nstate}, FT}(undef)
  l_M = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ξ1x3 = @scratch FT (Nq, Nq, Nqk) 3
  if dim == 3 || (dim == 2 && direction == EveryDirection)
    l_ξ2x1 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ2x2 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ2x3 = @scratch FT (Nq, Nq, Nqk) 3
  end
  if dim == 3 && direction == EveryDirection
    l_ξ3x1 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ3x2 = @scratch FT (Nq, Nq, Nqk) 3
    l_ξ3x3 = @scratch FT (Nq, Nq, Nqk) 3
  end

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      s_ω[j] = ω[j]
      @loop for i in (1:Nq; threadIdx().x)
        s_half_D[i, j] = D[i, j] / 2
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          l_M[i, j, k] = vgeo[ijk, _M, e]
          l_ξ1x1[i, j, k] = vgeo[ijk, _ξ1x1, e]
          l_ξ1x2[i, j, k] = vgeo[ijk, _ξ1x2, e]
          l_ξ1x3[i, j, k] = vgeo[ijk, _ξ1x3, e]
          if dim == 3 || (dim == 2 && direction == EveryDirection)
            l_ξ2x1[i, j, k] = vgeo[ijk, _ξ2x1, e]
            l_ξ2x2[i, j, k] = vgeo[ijk, _ξ2x2, e]
            l_ξ2x3[i, j, k] = vgeo[ijk, _ξ2x3, e]
          end
          if dim == 3 && direction == EveryDirection
            l_ξ3x1[i, j, k] = vgeo[ijk, _ξ3x1, e]
            l_ξ3x2[i, j, k] = vgeo[ijk, _ξ3x2, e]
            l_ξ3x3[i, j, k] = vgeo[ijk, _ξ3x3, e]
          end

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(FT)
          end

          @unroll for s = 1:nstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_nondiffusive!(bl, Grad{vars_state(bl,FT)}(l_F),
                             Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                             Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nstate
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          fill!(l_S, -zero(eltype(l_S)))
          source!(bl, Vars{vars_state(bl,FT)}(l_S),
                  Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                  Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] += l_S[s]
          end
        end
      end
    end
    @synchronize

    # Weak "outside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          @unroll for n = 1:Nq
            @unroll for s = 1:nstate
              Dni = s_half_D[n, i] * s_ω[n] / s_ω[i]
              if dim == 3 || (dim == 2 && direction == EveryDirection)
                Dnj = s_half_D[n, j] * s_ω[n] / s_ω[j]
              end
              if dim == 3 && direction == EveryDirection
                Dnk = s_half_D[n, k] * s_ω[n] / s_ω[k]
              end

              # ξ1-grid lines
              l_rhs[s, i, j, k] += l_ξ1x1[i, j, k] * Dni * s_F[1, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξ1x2[i, j, k] * Dni * s_F[2, n, j, k, s]
              l_rhs[s, i, j, k] += l_ξ1x3[i, j, k] * Dni * s_F[3, n, j, k, s]

              # ξ2-grid lines
              if dim == 3 || (dim == 2 && direction == EveryDirection)
                l_rhs[s, i, j, k] += l_ξ2x1[i, j, k] * Dnj * s_F[1, i, n, k, s]
                l_rhs[s, i, j, k] += l_ξ2x2[i, j, k] * Dnj * s_F[2, i, n, k, s]
                l_rhs[s, i, j, k] += l_ξ2x3[i, j, k] * Dnj * s_F[3, i, n, k, s]
              end

              # ξ3-grid lines
              if dim == 3 && direction == EveryDirection
                l_rhs[s, i, j, k] += l_ξ3x1[i, j, k] * Dnk * s_F[1, i, j, n, s]
                l_rhs[s, i, j, k] += l_ξ3x2[i, j, k] * Dnk * s_F[2, i, j, n, s]
                l_rhs[s, i, j, k] += l_ξ3x3[i, j, k] * Dnk * s_F[3, i, j, n, s]
              end
            end
          end
        end
      end
    end
    @synchronize

    # Add in the diffusive flux (multiply by 2 since derivative is halfed)
    # This allows symmetric treament of the 2nd order derivative terms
    # as well as build "inside metrics" flux
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))

          @unroll for s = 1:nviscstate
            l_Qvisc[s] = Qvisc[ijk, s, e]
          end
          
          @unroll for s = 1:nhyperviscstate
            l_Qhypervisc[s] = Qhypervisc_grad[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_diffusive!(bl, Grad{vars_state(bl,FT)}(l_F),
                          Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                          Vars{vars_diffusive(bl,FT)}(l_Qvisc),
                          Vars{vars_hyperdiffusive(bl,FT)}(l_Qhypervisc),
                          Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nstate
            F1, F2, F3 = s_F[1,i,j,k,s], s_F[2,i,j,k,s], s_F[3,i,j,k,s]

            F1 += 2l_F[1,s]
            F2 += 2l_F[2,s]
            F3 += 2l_F[3,s]

            s_F[1,i,j,k,s] = l_M[i, j, k] * (l_ξ1x1[i, j, k] * F1 +
                                              l_ξ1x2[i, j, k] * F2 +
                                              l_ξ1x3[i, j, k] * F3)
            if dim == 3 || (dim == 2 && direction == EveryDirection)
              s_F[2,i,j,k,s] = l_M[i, j, k] * (l_ξ2x1[i, j, k] * F1 +
                                               l_ξ2x2[i, j, k] * F2 +
                                               l_ξ2x3[i, j, k] * F3)
            end
            if dim == 3 && direction == EveryDirection
              s_F[3,i,j,k,s] = l_M[i, j, k] * (l_ξ3x1[i, j, k] * F1 +
                                               l_ξ3x2[i, j, k] * F2 +
                                               l_ξ3x3[i, j, k] * F3)
            end
          end
        end
      end
    end
    @synchronize

    # Weak "inside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          MI = vgeo[ijk, _MI, e]
          @unroll for s = 1:nstate
            @unroll for n = 1:Nq
              # ξ1-grid lines
              l_rhs[s, i, j, k] += MI * s_half_D[n, i] * s_F[1, n, j, k, s]

              # ξ2-grid lines
              if dim == 3 || (dim == 2 && direction == EveryDirection)
                l_rhs[s, i, j, k] += MI * s_half_D[n, j] * s_F[2, i, n, k, s]
              end

              # ξ3-grid lines
              if dim == 3 && direction == EveryDirection
                l_rhs[s, i, j, k] += MI * s_half_D[n, k] * s_F[3, i, j, n, s]
              end
            end
          end
        end
      end
    end
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:nstate
            rhs[ijk, s, e] = l_rhs[s, i, j, k]
          end
        end
      end
    end
    @synchronize
  end
  nothing
end

function volumerhs!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, ::VerticalDirection,
                    rhs, Q, Qvisc, Qhypervisc_grad, auxstate, vgeo, t,
                    ω, D, elems, increment) where {dim, polyorder}
  N = polyorder
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)
  
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nhyperviscstate = num_hyperdiffusive(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_F = @shmem FT (3, Nq, Nq, Nqk, nstate)
  s_ω = @shmem FT (Nq, )
  s_half_D = @shmem FT (Nq, Nq)
  l_rhs = @scratch FT (nstate, Nq, Nq, Nqk) 3

  source! !== nothing && (l_S = MArray{Tuple{nstate}, FT}(undef))
  l_Q = @scratch FT (nstate, Nq, Nq, Nqk) 3
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_Qhypervisc = MArray{Tuple{nhyperviscstate}, FT}(undef)
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3
  l_F = MArray{Tuple{3, nstate}, FT}(undef)
  l_M = @scratch FT (Nq, Nq, Nqk) 3

  l_ζx1 = @scratch FT (Nq, Nq, Nqk) 3
  l_ζx2 = @scratch FT (Nq, Nq, Nqk) 3
  l_ζx3 = @scratch FT (Nq, Nq, Nqk) 3

  _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
  _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
  _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      s_ω[j] = ω[j]
      @loop for i in (1:Nq; threadIdx().x)
        s_half_D[i, j] = D[i, j] / 2
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          l_M[i, j, k] = vgeo[ijk, _M, e]
          l_ζx1[i, j, k] = vgeo[ijk, _ζx1, e]
          l_ζx2[i, j, k] = vgeo[ijk, _ζx2, e]
          l_ζx3[i, j, k] = vgeo[ijk, _ζx3, e]

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] = increment ? rhs[ijk, s, e] : zero(FT)
          end

          @unroll for s = 1:nstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_nondiffusive!(bl, Grad{vars_state(bl,FT)}(l_F),
                             Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                             Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nstate
            s_F[1,i,j,k,s] = l_F[1,s]
            s_F[2,i,j,k,s] = l_F[2,s]
            s_F[3,i,j,k,s] = l_F[3,s]
          end

          # if source! !== nothing
          fill!(l_S, -zero(eltype(l_S)))
          source!(bl, Vars{vars_state(bl,FT)}(l_S),
                  Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                  Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nstate
            l_rhs[s, i, j, k] += l_S[s]
          end
        end
      end
    end
    @synchronize

    # Weak "outside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          @unroll for n = 1:Nq
            @unroll for s = 1:nstate
              if dim == 2
                Dnj = s_half_D[n, j] * s_ω[n] / s_ω[j]
                l_rhs[s, i, j, k] += l_ζx1[i, j, k] * Dnj * s_F[1, i, n, k, s]
                l_rhs[s, i, j, k] += l_ζx2[i, j, k] * Dnj * s_F[2, i, n, k, s]
                l_rhs[s, i, j, k] += l_ζx3[i, j, k] * Dnj * s_F[3, i, n, k, s]
              else
                Dnk = s_half_D[n, k] * s_ω[n] / s_ω[k]
                l_rhs[s, i, j, k] += l_ζx1[i, j, k] * Dnk * s_F[1, i, j, n, s]
                l_rhs[s, i, j, k] += l_ζx2[i, j, k] * Dnk * s_F[2, i, j, n, s]
                l_rhs[s, i, j, k] += l_ζx3[i, j, k] * Dnk * s_F[3, i, j, n, s]
              end
            end
          end
        end
      end
    end
    @synchronize

    # Build "inside metrics" flux
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))

          @unroll for s = 1:nviscstate
            l_Qvisc[s] = Qvisc[ijk, s, e]
          end
          
          @unroll for s = 1:nhyperviscstate
            l_Qhypervisc[s] = Qhypervisc_grad[ijk, s, e]
          end

          fill!(l_F, -zero(eltype(l_F)))
          flux_diffusive!(bl, Grad{vars_state(bl,FT)}(l_F),
                          Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                          Vars{vars_diffusive(bl,FT)}(l_Qvisc),
                          Vars{vars_hyperdiffusive(bl,FT)}(l_Qhypervisc),
                          Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

          @unroll for s = 1:nstate
            F1, F2, F3 = s_F[1,i,j,k,s], s_F[2,i,j,k,s], s_F[3,i,j,k,s]

            F1 += 2l_F[1,s]
            F2 += 2l_F[2,s]
            F3 += 2l_F[3,s]

            s_F[3,i,j,k,s] = l_M[i, j, k] * (l_ζx1[i, j, k] * F1 +
                                             l_ζx2[i, j, k] * F2 +
                                             l_ζx3[i, j, k] * F3)
          end
        end
      end
    end
    @synchronize

    # Weak "inside metrics" derivative
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          MI = vgeo[ijk, _MI, e]
          @unroll for s = 1:nstate
            @unroll for n = 1:Nq
              if dim == 2
                Dnj = s_half_D[n, j]
                l_rhs[s, i, j, k] += MI * Dnj * s_F[3, i, n, k, s]
              else
                Dnk = s_half_D[n, k]
                l_rhs[s, i, j, k] += MI * Dnk * s_F[3, i, j, n, s]
              end
            end
          end
        end
      end
    end
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:nstate
            rhs[ijk, s, e] = l_rhs[s, i, j, k]
          end
        end
      end
    end
    @synchronize
  end
  nothing
end

"""
    facerhs!(bl::BalanceLaw, Val(polyorder),
            numfluxnondiff::NumericalFluxNonDiffusive,
            numfluxdiff::NumericalFluxDiffusive,
            rhs, Q, Qvisc, auxstate,
            vgeo, sgeo, t, vmapM, vmapP, elemtobndy,
            elems)

Computational kernel: Evaluate the surface integrals on right-hand side of a
`BalanceLaw` semi-discretization.

See [`odefun!`](@ref) for usage.
"""
function facerhs!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, ::direction,
                  numfluxnondiff::NumericalFluxNonDiffusive,
                  numfluxdiff::NumericalFluxDiffusive,
                  rhs, Q, Qvisc, Qhypervisc_grad, auxstate, vgeo, sgeo, t, vmapM, vmapP,
                  elemtobndy, elems) where {dim, polyorder, direction}
  N = polyorder
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nhyperviscstate = num_hyperdiffusive(bl,FT)
  nauxstate = num_aux(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  faces = 1:nface
  if direction == VerticalDirection
    faces = nface-1:nface
  elseif direction == HorizontalDirection
    faces = 1:nface-2
  end

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  l_QM = MArray{Tuple{nstate}, FT}(undef)
  l_QviscM = MArray{Tuple{nviscstate}, FT}(undef)
  l_QhyperviscM = MArray{Tuple{nhyperviscstate}, FT}(undef)
  l_auxM = MArray{Tuple{nauxstate}, FT}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify QP
  l_QPnondiff = MArray{Tuple{nstate}, FT}(undef)
  l_QPdiff = MArray{Tuple{nstate}, FT}(undef)

  # Need two copies since numerical_flux_nondiffusive! can modify auxP
  l_auxPnondiff = MArray{Tuple{nauxstate}, FT}(undef)
  l_auxPdiff = MArray{Tuple{nauxstate}, FT}(undef)

  l_QviscP = MArray{Tuple{nviscstate}, FT}(undef)
  l_QhyperviscP = MArray{Tuple{nhyperviscstate}, FT}(undef)


  l_Q_bot1 = MArray{Tuple{nstate}, FT}(undef)
  l_Qvisc_bot1 = MArray{Tuple{nviscstate}, FT}(undef)
  l_aux_bot1 = MArray{Tuple{nauxstate}, FT}(undef)

  l_F = MArray{Tuple{nstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f in faces
      @loop for n in (1:Nfp; threadIdx().x)
        nM = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:nstate
          l_QM[s] = Q[vidM, s, eM]
        end

        @unroll for s = 1:nviscstate
          l_QviscM[s] = Qvisc[vidM, s, eM]
        end
        
        @unroll for s = 1:nhyperviscstate
          l_QhyperviscM[s] = Qhypervisc_grad[vidM, s, eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        # Load plus side data
        @unroll for s = 1:nstate
          l_QPdiff[s] = l_QPnondiff[s] = Q[vidP, s, eP]
        end

        @unroll for s = 1:nviscstate
          l_QviscP[s] = Qvisc[vidP, s, eP]
        end
        
        @unroll for s = 1:nhyperviscstate
          l_QhyperviscP[s] = Qhypervisc_grad[vidP, s, eP]
        end

        @unroll for s = 1:nauxstate
          l_auxPdiff[s] = l_auxPnondiff[s] = auxstate[vidP, s, eP]
        end

        bctype = elemtobndy[f, e]
        fill!(l_F, -zero(eltype(l_F)))
        if bctype == 0
          numerical_flux_nondiffusive!(numfluxnondiff, bl,
            Vars{vars_state(bl,FT)}(l_F), SVector(nM),
            Vars{vars_state(bl,FT)}(l_QM), Vars{vars_aux(bl,FT)}(l_auxM),
            Vars{vars_state(bl,FT)}(l_QPnondiff), Vars{vars_aux(bl,FT)}(l_auxPnondiff),
            t)
          numerical_flux_diffusive!(numfluxdiff, bl,
            Vars{vars_state(bl,FT)}(l_F), nM,
            Vars{vars_state(bl,FT)}(l_QM),
            Vars{vars_diffusive(bl,FT)}(l_QviscM),
            Vars{vars_hyperdiffusive(bl,FT)}(l_QhyperviscM),
            Vars{vars_aux(bl,FT)}(l_auxM),
            Vars{vars_state(bl,FT)}(l_QPdiff),
            Vars{vars_diffusive(bl,FT)}(l_QviscP),
            Vars{vars_hyperdiffusive(bl,FT)}(l_QhyperviscP),
            Vars{vars_aux(bl,FT)}(l_auxPdiff),
            t)
        else
          if (dim == 2 && f == 3) || (dim == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nstate
              l_Q_bot1[s] = Q[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nviscstate
              l_Qvisc_bot1[s] = Qvisc[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nauxstate
              l_aux_bot1[s] = auxstate[n + Nqk^2,s, e]
            end
          end
          numerical_boundary_flux_nondiffusive!(numfluxnondiff, bl,
            Vars{vars_state(bl,FT)}(l_F), SVector(nM),
            Vars{vars_state(bl,FT)}(l_QM), Vars{vars_aux(bl,FT)}(l_auxM),
            Vars{vars_state(bl,FT)}(l_QPnondiff), Vars{vars_aux(bl,FT)}(l_auxPnondiff),
            bctype, t,
            Vars{vars_state(bl,FT)}(l_Q_bot1), Vars{vars_aux(bl,FT)}(l_aux_bot1))
          numerical_boundary_flux_diffusive!(numfluxdiff, bl,
            Vars{vars_state(bl,FT)}(l_F), nM,
            Vars{vars_state(bl,FT)}(l_QM),
            Vars{vars_diffusive(bl,FT)}(l_QviscM),
            Vars{vars_hyperdiffusive(bl,FT)}(l_QhyperviscM),
            Vars{vars_aux(bl,FT)}(l_auxM),
            Vars{vars_state(bl,FT)}(l_QPdiff),
            Vars{vars_diffusive(bl,FT)}(l_QviscP),
            Vars{vars_hyperdiffusive(bl,FT)}(l_QhyperviscP),
            Vars{vars_aux(bl,FT)}(l_auxPdiff),
            bctype, t,
            Vars{vars_state(bl,FT)}(l_Q_bot1), Vars{vars_diffusive(bl,FT)}(l_Qvisc_bot1), Vars{vars_aux(bl,FT)}(l_aux_bot1))
        end

        #Update RHS
        @unroll for s = 1:nstate
          # FIXME: Should we pretch these?
          rhs[vidM, s, eM] -= vMI * sM * l_F[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function volumeviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                          ::direction, Q, Qvisc, Qhypervisc_grad, auxstate, vgeo, t, D,
                          elems) where {dim, polyorder, direction}
  N = polyorder

  FT = eltype(Q)
  nstate = num_state(bl,FT)
  ngradstate = num_gradient(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  ngradtransformstate = nstate

  s_G = @shmem FT (Nq, Nq, Nqk, ngradstate)
  s_D = @shmem FT (Nq, Nq)

  l_Q = @scratch FT (ngradtransformstate, Nq, Nq, Nqk) 3
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3
  l_G = MArray{Tuple{ngradstate}, FT}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_gradG = MArray{Tuple{3, ngradstate}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:ngradtransformstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          fill!(l_G, -zero(eltype(l_G)))
          gradvariables!(bl, Vars{vars_gradient(bl,FT)}(l_G), Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                     Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)
          @unroll for s = 1:ngradstate
            s_G[i, j, k, s] = l_G[s]
          end
        end
      end
    end
    @synchronize

    # Compute gradient of each state
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          ξ1x1, ξ1x2, ξ1x3 = vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]
          if dim == 3 || (dim == 2 && direction == EveryDirection)
            ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
          end
          if dim == 3 && direction == EveryDirection
            ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
          end

          @unroll for s = 1:ngradstate
            Gξ1 = Gξ2 = Gξ3 = zero(FT)
            @unroll for n = 1:Nq
              Gξ1 += s_D[i, n] * s_G[n, j, k, s]
              if dim == 3 || (dim == 2 && direction == EveryDirection)
                Gξ2 += s_D[j, n] * s_G[i, n, k, s]
              end
              if dim == 3 && direction == EveryDirection
                Gξ3 += s_D[k, n] * s_G[i, j, n, s]
              end
            end
            l_gradG[1, s] = ξ1x1 * Gξ1
            l_gradG[2, s] = ξ1x2 * Gξ1
            l_gradG[3, s] = ξ1x3 * Gξ1

            if dim == 3 || (dim == 2 && direction == EveryDirection)
              l_gradG[1, s] += ξ2x1 * Gξ2
              l_gradG[2, s] += ξ2x2 * Gξ2
              l_gradG[3, s] += ξ2x3 * Gξ2
            end

            if dim == 3 && direction == EveryDirection
              l_gradG[1, s] += ξ3x1 * Gξ3
              l_gradG[2, s] += ξ3x2 * Gξ3
              l_gradG[3, s] += ξ3x3 * Gξ3
            end
          end

          @unroll for s = 1:ngradlapstate
            Qhypervisc_grad[ijk, 3(s - 1) + 1, e] = l_gradG[1, s]
            Qhypervisc_grad[ijk, 3(s - 1) + 2, e] = l_gradG[2, s]
            Qhypervisc_grad[ijk, 3(s - 1) + 3, e] = l_gradG[3, s]
          end

          if nviscstate > 0
            fill!(l_Qvisc, -zero(eltype(l_Qvisc)))
            diffusive!(bl, Vars{vars_diffusive(bl,FT)}(l_Qvisc), Grad{vars_gradient(bl,FT)}(l_gradG),
                       Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]), Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)

            @unroll for s = 1:nviscstate
              Qvisc[ijk, s, e] = l_Qvisc[s]
            end
          end
        end
      end
    end
    @synchronize
  end
end

function volumeviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                          ::VerticalDirection, Q, Qvisc, Qhypervisc_grad, auxstate, vgeo, t, D,
                          elems) where {dim, polyorder}
  N = polyorder

  FT = eltype(Q)
  nstate = num_state(bl,FT)
  ngradstate = num_gradient(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  ngradtransformstate = nstate

  s_G = @shmem FT (Nq, Nq, Nqk, ngradstate)
  s_D = @shmem FT (Nq, Nq)

  l_Q = @scratch FT (ngradtransformstate, Nq, Nq, Nqk) 3
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3
  l_G = MArray{Tuple{ngradstate}, FT}(undef)
  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_gradG = MArray{Tuple{3, ngradstate}, FT}(undef)

  _ζx1 = dim == 2 ? _ξ2x1 : _ξ3x1
  _ζx2 = dim == 2 ? _ξ2x2 : _ξ3x2
  _ζx3 = dim == 2 ? _ξ2x3 : _ξ3x3

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:ngradtransformstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          fill!(l_G, -zero(eltype(l_G)))
          gradvariables!(bl, Vars{vars_gradient(bl,FT)}(l_G),
                         Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                         Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)
          @unroll for s = 1:ngradstate
            s_G[i, j, k, s] = l_G[s]
          end
        end
      end
    end
    @synchronize

    # Compute gradient of each state
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          ζx1 = vgeo[ijk, _ζx1, e]
          ζx2 = vgeo[ijk, _ζx2, e]
          ζx3 = vgeo[ijk, _ζx3, e]

          @unroll for s = 1:ngradstate
            Gζ = zero(FT)
            @unroll for n = 1:Nq
              if dim == 2
                Gζ += s_D[j, n] * s_G[i, n, k, s]
              elseif dim == 3
                Gζ += s_D[k, n] * s_G[i, j, n, s]
              end
            end
            l_gradG[1, s] = ζx1 * Gζ
            l_gradG[2, s] = ζx2 * Gζ
            l_gradG[3, s] = ζx3 * Gζ
          end

          @unroll for s = 1:ngradlapstate
            Qhypervisc_grad[ijk, 3(s - 1) + 1, e] = l_gradG[1, s]
            Qhypervisc_grad[ijk, 3(s - 1) + 2, e] = l_gradG[2, s]
            Qhypervisc_grad[ijk, 3(s - 1) + 3, e] = l_gradG[3, s]
          end

          if nviscstate > 0
            fill!(l_Qvisc, -zero(eltype(l_Qvisc)))
            diffusive!(bl, Vars{vars_diffusive(bl,FT)}(l_Qvisc),
                       Grad{vars_gradient(bl,FT)}(l_gradG),
                       Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                       Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]), t)
          end

          @unroll for s = 1:nviscstate
            Qvisc[ijk, s, e] = l_Qvisc[s]
          end
        end
      end
    end
    @synchronize
  end
end

function faceviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                        ::direction,
                        gradnumpenalty::GradNumericalPenalty,
                        Q, Qvisc, Qhypervisc_grad, auxstate, vgeo, sgeo, t, vmapM, vmapP,
                        elemtobndy, elems) where {dim, polyorder, direction}
  N = polyorder
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  ngradstate = num_gradient(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  faces = 1:nface
  if direction == VerticalDirection
    faces = nface-1:nface
  elseif direction == HorizontalDirection
    faces = 1:nface-2
  end

  Nqk = dim == 2 ? 1 : N+1

  ngradtransformstate = nstate

  l_QM = MArray{Tuple{ngradtransformstate}, FT}(undef)
  l_auxM = MArray{Tuple{nauxstate}, FT}(undef)
  l_GM = MArray{Tuple{ngradstate}, FT}(undef)

  l_QP = MArray{Tuple{ngradtransformstate}, FT}(undef)
  l_auxP = MArray{Tuple{nauxstate}, FT}(undef)
  l_GP = MArray{Tuple{ngradstate}, FT}(undef)

  l_Qvisc = MArray{Tuple{nviscstate}, FT}(undef)
  l_gradG = MArray{Tuple{3, ngradstate}, FT}(undef)

  l_Q_bot1 = MArray{Tuple{nstate}, FT}(undef)
  l_aux_bot1 = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f in faces
      @loop for n in (1:Nfp; threadIdx().x)
        nM = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:ngradtransformstate
          l_QM[s] = Q[vidM, s, eM]
        end

        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        fill!(l_GM, -zero(eltype(l_GM)))
        gradvariables!(bl, Vars{vars_gradient(bl,FT)}(l_GM),
                       Vars{vars_state(bl,FT)}(l_QM),
                       Vars{vars_aux(bl,FT)}(l_auxM), t)

        # Load plus side data
        @unroll for s = 1:ngradtransformstate
          l_QP[s] = Q[vidP, s, eP]
        end

        @unroll for s = 1:nauxstate
          l_auxP[s] = auxstate[vidP, s, eP]
        end

        fill!(l_GP, -zero(eltype(l_GP)))
        gradvariables!(bl, Vars{vars_gradient(bl,FT)}(l_GP),
                       Vars{vars_state(bl,FT)}(l_QP),
                       Vars{vars_aux(bl,FT)}(l_auxP), t)

        bctype = elemtobndy[f, e]
        fill!(l_Qvisc, -zero(eltype(l_Qvisc)))
        if bctype == 0
          gradient_penalty!(gradnumpenalty, bl,
            Grad{vars_gradient(bl, FT)}(l_gradG), SVector(nM),
            Vars{vars_gradient(bl,FT)}(l_GM), Vars{vars_state(bl,FT)}(l_QM), Vars{vars_aux(bl,FT)}(l_auxM),
            Vars{vars_gradient(bl,FT)}(l_GP), Vars{vars_state(bl,FT)}(l_QP), Vars{vars_aux(bl,FT)}(l_auxP),
            t)
          if nviscstate > 0
            diffusive!(bl, Vars{vars_diffusive(bl,FT)}(l_Qvisc),
                       Grad{vars_gradient(bl,FT)}(l_gradG),
                       Vars{vars_state(bl,FT)}(l_QM), Vars{vars_aux(bl,FT)}(l_auxM),
                       t)
          end
        else
          if (dim == 2 && f == 3) || (dim == 3 && f == 5)
            # Loop up the first element along all horizontal elements
            @unroll for s = 1:nstate
              l_Q_bot1[s] = Q[n + Nqk^2, s, e]
            end
            @unroll for s = 1:nauxstate
              l_aux_bot1[s] = auxstate[n + Nqk^2,s, e]
            end
          end
          gradient_boundary_penalty!(gradnumpenalty, bl,
            Grad{vars_gradient(bl, FT)}(l_gradG), SVector(nM),
            Vars{vars_gradient(bl,FT)}(l_GM), Vars{vars_state(bl,FT)}(l_QM), Vars{vars_aux(bl,FT)}(l_auxM),
            Vars{vars_gradient(bl,FT)}(l_GP), Vars{vars_state(bl,FT)}(l_QP), Vars{vars_aux(bl,FT)}(l_auxP),
            bctype, t, Vars{vars_state(bl,FT)}(l_Q_bot1), Vars{vars_aux(bl,FT)}(l_aux_bot1))
          if nviscstate > 0
            diffusive!(bl, Vars{vars_diffusive(bl,FT)}(l_Qvisc),
                       Grad{vars_gradient(bl,FT)}(l_gradG),
                       Vars{vars_state(bl,FT)}(l_QM), Vars{vars_aux(bl,FT)}(l_auxM),
                       t)
          end
        end

        @unroll for s = 1:ngradlapstate
          Qhypervisc_grad[vidM, 3(s - 1) + 1, eM] += vMI * sM * l_gradG[1, s]
          Qhypervisc_grad[vidM, 3(s - 1) + 2, eM] += vMI * sM * l_gradG[2, s]
          Qhypervisc_grad[vidM, 3(s - 1) + 3, eM] += vMI * sM * l_gradG[3, s]
        end
        
        @unroll for s = 1:nviscstate
          Qvisc[vidM, s, eM] += vMI * sM * l_Qvisc[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function initstate!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, state, auxstate, vgeo, elems, args...) where {dim, polyorder}
  N = polyorder
  FT = eltype(auxstate)
  nauxstate = num_aux(bl,FT)
  nstate = num_state(bl,FT)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_state = MArray{Tuple{nstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      coords = SVector(vgeo[n, _x1, e], vgeo[n, _x2, e], vgeo[n, _x3, e])
      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end
      @unroll for s = 1:nstate
        l_state[s] = state[n, s, e]
      end
      init_state!(bl, Vars{vars_state(bl,FT)}(l_state), Vars{vars_aux(bl,FT)}(l_aux), coords, args...)
      @unroll for s = 1:nstate
        state[n, s, e] = l_state[s]
      end
    end
  end
end


"""
    initauxstate!(bl::BalanceLaw, Val(polyorder), auxstate, vgeo, elems)

Computational kernel: Initialize the auxiliary state

See [`DGBalanceLaw`](@ref) for usage.
"""
function initauxstate!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder}, auxstate, vgeo, elems) where {dim, polyorder}
  N = polyorder
  FT = eltype(auxstate)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq
  Np = Nq * Nq * Nqk

  l_aux = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      init_aux!(bl, Vars{vars_aux(bl,FT)}(l_aux), LocalGeometry(Val(polyorder),vgeo,n,e))

      @unroll for s = 1:nauxstate
        auxstate[n, s, e] = l_aux[s]
      end
    end
  end
end

"""
    knl_nodal_update_aux!(bl::BalanceLaw, ::Val{dim}, ::Val{N}, f!, Q, auxstate,
                          t, elems) where {dim, N}

Update the auxiliary state array
"""
function knl_nodal_update_aux!(bl::BalanceLaw, ::Val{dim}, ::Val{N}, f!, Q,
                               auxstate, t, elems) where {dim, N}
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Q = MArray{Tuple{nstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nstate
        l_Q[s] = Q[n, s, e]
      end

      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      f!(bl,
         Vars{vars_state(bl,FT)}(l_Q),
         Vars{vars_aux(bl,FT)}(l_aux),
         t)

      @unroll for s = 1:nauxstate
        auxstate[n, s, e] = l_aux[s]
      end
    end
  end
end

"""
    knl_nodal_update_aux!(bl::BalanceLaw, ::Val{dim}, ::Val{N}, f!, Q, auxstate, diffstate,
                          t, elems) where {dim, N}

Update the auxiliary state array
"""
function knl_nodal_update_aux!(bl::BalanceLaw, ::Val{dim}, ::Val{N}, f!, Q,
                               auxstate, diffstate, t, elems) where {dim, N}
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  nviscstate = num_diffusive(bl,FT)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  Np = Nq * Nq * Nqk

  l_Q = MArray{Tuple{nstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)
  l_diff = MArray{Tuple{nviscstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    @loop for n in (1:Np; threadIdx().x)
      @unroll for s = 1:nstate
        l_Q[s] = Q[n, s, e]
      end

      @unroll for s = 1:nauxstate
        l_aux[s] = auxstate[n, s, e]
      end

      @unroll for s = 1:nviscstate
        l_diff[s] = diffstate[n, s, e]
      end

      f!(bl,
         Vars{vars_state(bl,FT)}(l_Q),
         Vars{vars_aux(bl,FT)}(l_aux),
         Vars{vars_diffusive(bl,FT)}(l_diff),
         t)

      @unroll for s = 1:nauxstate
        auxstate[n, s, e] = l_aux[s]
      end
    end
  end
end

"""
    knl_indefinite_stack_integral!(::Val{dim}, ::Val{N}, ::Val{nstate},
                                            ::Val{nauxstate}, ::Val{nvertelem},
                                            int_knl!, Q, auxstate, vgeo, Imat,
                                            elems, ::Val{outstate}
                                           ) where {dim, N, nstate, nauxstate,
                                                    outstate, nvertelem}

Computational kernel: compute indefinite integral along the vertical stack

See [`DGBalanceLaw`](@ref) for usage.
"""
function knl_indefinite_stack_integral!(bl::BalanceLaw, ::Val{dim}, ::Val{N}, ::Val{nvertelem},
                                        Q, auxstate, vgeo, Imat,
                                        elems, ::Val{nout}
                                       ) where {dim, N, nvertelem,
                                                nout}
  FT = eltype(Q)
  nstate = num_state(bl,FT)
  nauxstate = num_aux(bl,FT)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  l_Q = MArray{Tuple{nstate}, FT}(undef)
  l_aux = MArray{Tuple{nauxstate}, FT}(undef)
  l_knl = MArray{Tuple{nout, Nq}, FT}(undef)
  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_int = @scratch FT (nout, Nq, Nq, Nqj) 2

  s_I = @shmem FT (Nq, Nq)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for i in (1:Nq; threadIdx().x)
      @unroll for n = 1:Nq
        s_I[i, n] = Imat[i, n]
      end
    end
  end
  @synchronize

  @inbounds @loop for eh in (elems; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for k in 1:Nq
          @unroll for s = 1:nout
            l_int[s, k, i, j] = 0
          end
        end
      end
    end
    # Loop up the stack of elements
    for ev = 1:nvertelem
      e = ev + (eh - 1) * nvertelem

      # Evaluate the integral kernel at each DOF in the slabk
      @loop for j in (1:Nqj; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          # loop up the pencil
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            Jc = vgeo[ijk, _JcV, e]
            @unroll for s = 1:nstate
              l_Q[s] = Q[ijk, s, e]
            end

            @unroll for s = 1:nauxstate
              l_aux[s] = auxstate[ijk, s, e]
            end

            integrate_aux!(bl, Vars{vars_integrals(bl, FT)}(view(l_knl, :, k)),
              Vars{vars_state(bl, FT)}(l_Q), Vars{vars_aux(bl,FT)}(l_aux))

            # multiply in the curve jacobian
            @unroll for s = 1:nout
              l_knl[s, k] *= Jc
            end
          end

          # Evaluate the integral up the element
          @unroll for s = 1:nout
            @unroll for k in 1:Nq
              @unroll for n in 1:Nq
                l_int[s, k, i, j] += s_I[k, n] * l_knl[s, n]
              end
            end
          end

          # Store out to memory and reset the background value for next element
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            @unroll for ind_out = 1:nout
              auxstate[ijk, ind_out, e] = l_int[ind_out, k, i, j]
              l_int[ind_out, k, i, j] = l_int[ind_out, Nq, i, j]
            end
          end
        end
      end
    end
  end
  nothing
end

function knl_reverse_indefinite_stack_integral!(::Val{dim}, ::Val{N},
                                                ::Val{nvertelem}, auxstate, elems,
                                                ::Val{nout}
                                               ) where {dim, N, nvertelem,
                                                        nout}
  FT = eltype(auxstate)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  l_T = MArray{Tuple{nout}, FT}(undef)
  l_V = MArray{Tuple{nout}, FT}(undef)

  @inbounds @loop for eh in (elems; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        ijk = i + Nq * ((j-1) + Nqj * (Nq-1))
        et = nvertelem + (eh - 1) * nvertelem
        @unroll for s = 1:nout
          l_T[s] = auxstate[ijk, s, et]
        end

        # Loop up the stack of elements
        for ev = 1:nvertelem
          e = ev + (eh - 1) * nvertelem
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            @unroll for s = 1:nout
              l_V[s] = auxstate[ijk, s, e]
            end
            @unroll for s = 1:nout
              auxstate[ijk, nout+s, e] = l_T[s] - l_V[s]
            end
          end
        end
      end
    end
  end
  nothing
end

# TODO: Generalize to more than one field?
function knl_copy_stack_field_down!(::Val{dim}, ::Val{N}, ::Val{nvertelem},
                                    auxstate, elems, ::Val{fldin},
                                    ::Val{fldout}) where {dim, N, nvertelem,
                                                          fldin, fldout}
  DFloat = eltype(auxstate)

  Nq = N + 1
  Nqj = dim == 2 ? 1 : Nq

  # note that k is the second not 4th index (since this is scratch memory and k
  # needs to be persistent across threads)
  @inbounds @loop for eh in (elems; blockIdx().x)
    # Initialize the constant state at zero
    @loop for j in (1:Nqj; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        ijk = i + Nq * ((j-1) + Nqj * (Nq-1))
        et = nvertelem + (eh - 1) * nvertelem
        val = auxstate[ijk, fldin, et]

        # Loop up the stack of elements
        for ev = 1:nvertelem
          e = ev + (eh - 1) * nvertelem
          @unroll for k in 1:Nq
            ijk = i + Nq * ((j-1) + Nqj * (k-1))
            auxstate[ijk, fldout, e] = val
          end
        end
      end
    end
  end
  nothing
end

function volumedivgrad!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                        ::direction, Qhypervisc_grad, Qhypervisc_div, vgeo, D,
                        elems) where {dim, polyorder, direction}
  N = polyorder
  FT = eltype(Qhypervisc_grad)
  ngradlapstate = num_gradient_laplacian(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_grad = @shmem FT (Nq, Nq, Nqk, ngradlapstate, 3)
  s_D = @shmem FT (Nq, Nq)

  l_div = MArray{Tuple{ngradlapstate}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:ngradlapstate
            s_grad[i, j, k, s, 1] = Qhypervisc_grad[ijk, 3(s - 1) + 1, e]
            s_grad[i, j, k, s, 2] = Qhypervisc_grad[ijk, 3(s - 1) + 2, e]
            s_grad[i, j, k, s, 3] = Qhypervisc_grad[ijk, 3(s - 1) + 3, e]
          end
        end
      end
    end
    @synchronize

    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          ξ1x1, ξ1x2, ξ1x3 = vgeo[ijk, _ξ1x1, e], vgeo[ijk, _ξ1x2, e], vgeo[ijk, _ξ1x3, e]
          if dim == 3 || (dim == 2 && direction == EveryDirection)
            ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
          end
          if dim == 3 && direction == EveryDirection
            ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
          end

          @unroll for s = 1:ngradlapstate
            g1ξ1 = g1ξ2 = g1ξ3 = zero(FT)
            g2ξ1 = g2ξ2 = g2ξ3 = zero(FT)
            g3ξ1 = g3ξ2 = g3ξ3 = zero(FT)
            @unroll for n = 1:Nq
              Din = s_D[i, n] 
              g1ξ1 += Din * s_grad[n, j, k, s, 1]
              g2ξ1 += Din * s_grad[n, j, k, s, 2]
              g3ξ1 += Din * s_grad[n, j, k, s, 3]
              if dim == 3 || (dim == 2 && direction == EveryDirection)
                Djn = s_D[j, n] 
                g1ξ2 += Djn * s_grad[i, n, k, s, 1]
                g2ξ2 += Djn * s_grad[i, n, k, s, 2]
                g3ξ2 += Djn * s_grad[i, n, k, s, 3]
              end
              if dim == 3 && direction == EveryDirection
                Dkn = s_D[k, n] 
                g1ξ3 += Dkn * s_grad[i, j, n, s, 1]
                g2ξ3 += Dkn * s_grad[i, j, n, s, 2]
                g3ξ3 += Dkn * s_grad[i, j, n, s, 3]
              end
            end
            l_div[s] = ξ1x1 * g1ξ1 + ξ1x2 * g2ξ1 + ξ1x3 * g3ξ1

            if dim == 3 || (dim == 2 && direction == EveryDirection)
              l_div[s] += ξ2x1 * g1ξ2 + ξ2x2 * g2ξ2 + ξ2x3 * g3ξ2
            end

            if dim == 3 && direction == EveryDirection
              l_div[s] += ξ3x1 * g1ξ3 + ξ3x2 * g2ξ3 + ξ3x3 * g3ξ3
            end
          end

          @unroll for s = 1:ngradlapstate
            Qhypervisc_div[ijk, s, e] = l_div[s]
          end
        end
      end
    end
    @synchronize
  end
end

function volumedivgrad!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                        ::VerticalDirection, Qhypervisc_grad, Qhypervisc_div, vgeo, D,
                        elems) where {dim, polyorder}
  N = polyorder
  FT = eltype(Qhypervisc_grad)
  ngradlapstate = num_gradient_laplacian(bl,FT)

  Nq = N + 1

  Nqk = dim == 2 ? 1 : Nq

  s_grad = @shmem FT (Nq, Nq, Nqk, ngradlapstate, 3)
  s_D = @shmem FT (Nq, Nq)

  l_div = MArray{Tuple{ngradlapstate}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:ngradlapstate
            s_grad[i, j, k, s, 1] = Qhypervisc_grad[ijk, 3(s - 1) + 1, e]
            s_grad[i, j, k, s, 2] = Qhypervisc_grad[ijk, 3(s - 1) + 2, e]
            s_grad[i, j, k, s, 3] = Qhypervisc_grad[ijk, 3(s - 1) + 3, e]
          end
        end
      end
    end
    @synchronize

    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          if dim == 2
            ξ2x1, ξ2x2, ξ2x3 = vgeo[ijk, _ξ2x1, e], vgeo[ijk, _ξ2x2, e], vgeo[ijk, _ξ2x3, e]
          else
            ξ3x1, ξ3x2, ξ3x3 = vgeo[ijk, _ξ3x1, e], vgeo[ijk, _ξ3x2, e], vgeo[ijk, _ξ3x3, e]
          end

          @unroll for s = 1:ngradlapstate
            g1ξv = g2ξv = g3ξv = zero(FT)
            @unroll for n = 1:Nq
              if dim == 2
                Djn = s_D[j, n] 
                g1ξv += Djn * s_grad[i, n, k, s, 1]
                g2ξv += Djn * s_grad[i, n, k, s, 2]
                g3ξv += Djn * s_grad[i, n, k, s, 3]
              else
                Dkn = s_D[k, n] 
                g1ξv += Dkn * s_grad[i, j, n, s, 1]
                g2ξv += Dkn * s_grad[i, j, n, s, 2]
                g3ξv += Dkn * s_grad[i, j, n, s, 3]
              end
            end

            if dim == 2
              l_div[s] = ξ2x1 * g1ξv + ξ2x2 * g2ξv + ξ2x3 * g3ξv
            else
              l_div[s] = ξ3x1 * g1ξv + ξ3x2 * g2ξv + ξ3x3 * g3ξv
            end
          end

          @unroll for s = 1:ngradlapstate
            Qhypervisc_div[ijk, s, e] = l_div[s]
          end
        end
      end
    end
    @synchronize
  end
end

function facedivgrad!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                      ::direction,
                      divgradnumpenalty,
                      Qhypervisc_grad, Qhypervisc_div, vgeo, sgeo, vmapM, vmapP,
                      elemtobndy, elems) where {dim, polyorder, direction}
  N = polyorder
  FT = eltype(Qhypervisc_grad)
  ngradlapstate = num_gradient_laplacian(bl,FT)

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  faces = 1:nface
  if direction == VerticalDirection
    faces = nface-1:nface
  elseif direction == HorizontalDirection
    faces = 1:nface-2
  end

  Nqk = dim == 2 ? 1 : N+1

  l_gradM = MArray{Tuple{ngradlapstate, 3}, FT}(undef)
  l_gradP = MArray{Tuple{ngradlapstate, 3}, FT}(undef)
  l_div = MArray{Tuple{ngradlapstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f in faces
      @loop for n in (1:Nfp; threadIdx().x)
        nM = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:ngradlapstate
          l_gradM[s, 1] = Qhypervisc_grad[vidM, 3(s - 1) + 1, eM]
          l_gradM[s, 2] = Qhypervisc_grad[vidM, 3(s - 1) + 2, eM]
          l_gradM[s, 3] = Qhypervisc_grad[vidM, 3(s - 1) + 3, eM]
        end

        # Load plus side data
        @unroll for s = 1:ngradlapstate
          l_gradP[s, 1] = Qhypervisc_grad[vidP, 3(s - 1) + 1, eP]
          l_gradP[s, 2] = Qhypervisc_grad[vidP, 3(s - 1) + 2, eP]
          l_gradP[s, 3] = Qhypervisc_grad[vidP, 3(s - 1) + 3, eP]
        end

        bctype = elemtobndy[f, e]
        if bctype == 0
          divergence_penalty!(divgradnumpenalty, bl, l_div,
                              nM, l_gradM, l_gradP)
        else
          divergence_boundary_penalty!(divgradnumpenalty, bl, l_div,
                                       nM, l_gradM, l_gradP, bctype)
        end

        @unroll for s = 1:ngradlapstate
          Qhypervisc_div[vidM, s, eM] += vMI * sM * l_div[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end

function volumehyperviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                               ::direction,
                               Qhypervisc_grad, Qhypervisc_div,
                               Q, auxstate,
                               vgeo, D, elems, t) where {dim, polyorder, direction}
  N = polyorder

  FT = eltype(Qhypervisc_grad)
  nstate = num_state(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nhyperviscstate = num_hyperdiffusive(bl,FT)
  nauxstate = num_aux(bl,FT)
  ngradtransformstate = nstate

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  s_lap = @shmem FT (Nq, Nq, Nqk, ngradlapstate)
  s_D = @shmem FT (Nq, Nq)
  l_Q = @scratch FT (ngradtransformstate, Nq, Nq, Nqk) 3
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3

  l_grad_lap = MArray{Tuple{3, ngradlapstate}, FT}(undef)
  l_Qhypervisc = MArray{Tuple{nhyperviscstate}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          
          @unroll for s = 1:ngradtransformstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          @unroll for s = 1:ngradlapstate
            s_lap[i, j, k, s] = Qhypervisc_div[ijk, s, e]
          end
        end
      end
    end
    @synchronize

    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))

          @unroll for s = 1:ngradlapstate
            lap_ξ1x1 = lap_ξ2x1 = lap_ξ3x1 = zero(FT)
            lap_ξ1x2 = lap_ξ2x2 = lap_ξ3x2 = zero(FT)
            lap_ξ1x3 = lap_ξ2x3 = lap_ξ3x3 = zero(FT)
            @unroll for n = 1:Nq
              njk = n + Nq * ((j-1) + Nq * (k-1))
              M = vgeo[njk, _M, e]
              ξ1x1 = vgeo[njk, _ξ1x1, e]
              ξ1x2 = vgeo[njk, _ξ1x2, e]
              ξ1x3 = vgeo[njk, _ξ1x3, e]
              Dni = s_D[n, i]
              lap_njk = s_lap[n, j, k, s] 
              lap_ξ1x1 += ξ1x1 * M * Dni * lap_njk
              lap_ξ1x2 += ξ1x2 * M * Dni * lap_njk
              lap_ξ1x3 += ξ1x3 * M * Dni * lap_njk
              if dim == 3 || (dim == 2 && direction == EveryDirection)
                ink = i + Nq * ((n-1) + Nq * (k-1))
                M = vgeo[ink, _M, e]
                ξ2x1 = vgeo[ink, _ξ2x1, e]
                ξ2x2 = vgeo[ink, _ξ2x2, e]
                ξ2x3 = vgeo[ink, _ξ2x3, e]
                Dnj = s_D[n, j]
                lap_ink = s_lap[i, n, k, s] 
                lap_ξ2x1 += ξ2x1 * M * Dnj * lap_ink
                lap_ξ2x2 += ξ2x2 * M * Dnj * lap_ink
                lap_ξ2x3 += ξ2x3 * M * Dnj * lap_ink
              end
              if dim == 3 && direction == EveryDirection
                ijn = i + Nq * ((j-1) + Nq * (n-1))
                M = vgeo[ijn, _M, e]
                ξ3x1 = vgeo[ijn, _ξ3x1, e]
                ξ3x2 = vgeo[ijn, _ξ3x2, e]
                ξ3x3 = vgeo[ijn, _ξ3x3, e]
                Dnk = s_D[n, k]
                lap_ijn = s_lap[i, j, n, s] 
                lap_ξ3x1 += ξ3x1 * M * Dnk * lap_ijn
                lap_ξ3x2 += ξ3x2 * M * Dnk * lap_ijn
                lap_ξ3x3 += ξ3x3 * M * Dnk * lap_ijn
              end
            end
            
            MI = vgeo[ijk, _MI, e]
            l_grad_lap[1, s] = -MI * lap_ξ1x1
            l_grad_lap[2, s] = -MI * lap_ξ1x2
            l_grad_lap[3, s] = -MI * lap_ξ1x3

            if dim == 3 || (dim == 2 && direction == EveryDirection)
              l_grad_lap[1, s] -= MI * lap_ξ2x1
              l_grad_lap[2, s] -= MI * lap_ξ2x2
              l_grad_lap[3, s] -= MI * lap_ξ2x3
            end

            if dim == 3 && direction == EveryDirection
              l_grad_lap[1, s] -= MI * lap_ξ3x1
              l_grad_lap[2, s] -= MI * lap_ξ3x2
              l_grad_lap[3, s] -= MI * lap_ξ3x3
            end
          end

          fill!(l_Qhypervisc, -zero(eltype(l_Qhypervisc)))
          hyperdiffusive!(bl,
                          Vars{vars_hyperdiffusive(bl,FT)}(l_Qhypervisc),
                          Grad{vars_gradient_laplacian(bl,FT)}(l_grad_lap),
                          Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                          Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]),
                          t)
          @unroll for s = 1:nhyperviscstate
            Qhypervisc_grad[ijk, s, e] = l_Qhypervisc[s]
          end
        end
      end
    end
    @synchronize
  end
end

function volumehyperviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                               ::VerticalDirection,
                               Qhypervisc_grad, Qhypervisc_div,
                               Q, auxstate,
                               vgeo, D, elems, t) where {dim, polyorder}
  N = polyorder

  FT = eltype(Qhypervisc_grad)
  nstate = num_state(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nhyperviscstate = num_hyperdiffusive(bl,FT)
  nauxstate = num_aux(bl,FT)
  ngradtransformstate = nstate

  Nq = N + 1
  Nqk = dim == 2 ? 1 : Nq

  s_lap = @shmem FT (Nq, Nq, Nqk, ngradlapstate)
  s_D = @shmem FT (Nq, Nq)
  l_Q = @scratch FT (ngradtransformstate, Nq, Nq, Nqk) 3
  l_aux = @scratch FT (nauxstate, Nq, Nq, Nqk) 3

  l_grad_lap = MArray{Tuple{3, ngradlapstate}, FT}(undef)
  l_Qhypervisc = MArray{Tuple{nhyperviscstate}, FT}(undef)

  @inbounds @loop for k in (1; threadIdx().z)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        s_D[i, j] = D[i, j]
      end
    end
  end

  @inbounds @views @loop for e in (elems; blockIdx().x)
    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          
          @unroll for s = 1:ngradtransformstate
            l_Q[s, i, j, k] = Q[ijk, s, e]
          end

          @unroll for s = 1:nauxstate
            l_aux[s, i, j, k] = auxstate[ijk, s, e]
          end

          @unroll for s = 1:ngradlapstate
            s_lap[i, j, k, s] = Qhypervisc_div[ijk, s, e]
          end
        end
      end
    end
    @synchronize

    @loop for k in (1:Nqk; threadIdx().z)
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)
          ijk = i + Nq * ((j-1) + Nq * (k-1))
          @unroll for s = 1:ngradlapstate
            lap_ξvx1 = lap_ξvx2 = lap_ξvx3 = zero(FT)
            @unroll for n = 1:Nq
              if dim == 2
                ink = i + Nq * ((n-1) + Nq * (k-1))
                M = vgeo[ink, _M, e]
                ξ2x1 = vgeo[ink, _ξ2x1, e]
                ξ2x2 = vgeo[ink, _ξ2x2, e]
                ξ2x3 = vgeo[ink, _ξ2x3, e]
                Dnj = s_D[n, j]
                lap_ink = s_lap[i, n, k, s] 
                lap_ξvx1 += ξ2x1 * M * Dnj * lap_ink
                lap_ξvx2 += ξ2x2 * M * Dnj * lap_ink
                lap_ξvx3 += ξ2x3 * M * Dnj * lap_ink
              else
                ijn = i + Nq * ((j-1) + Nq * (n-1))
                M = vgeo[ijn, _M, e]
                ξ3x1 = vgeo[ijn, _ξ3x1, e]
                ξ3x2 = vgeo[ijn, _ξ3x2, e]
                ξ3x3 = vgeo[ijn, _ξ3x3, e]
                Dnk = s_D[n, k]
                lap_ijn = s_lap[i, j, n, s] 
                lap_ξvx1 += ξ3x1 * M * Dnk * lap_ijn
                lap_ξvx2 += ξ3x2 * M * Dnk * lap_ijn
                lap_ξvx3 += ξ3x3 * M * Dnk * lap_ijn
              end
            end
            
            MI = vgeo[ijk, _MI, e]
            l_grad_lap[1, s] = -MI * lap_ξvx1
            l_grad_lap[2, s] = -MI * lap_ξvx2
            l_grad_lap[3, s] = -MI * lap_ξvx3
          end

          fill!(l_Qhypervisc, -zero(eltype(l_Qhypervisc)))
          hyperdiffusive!(bl,
                          Vars{vars_hyperdiffusive(bl,FT)}(l_Qhypervisc),
                          Grad{vars_gradient_laplacian(bl,FT)}(l_grad_lap),
                          Vars{vars_state(bl,FT)}(l_Q[:, i, j, k]),
                          Vars{vars_aux(bl,FT)}(l_aux[:, i, j, k]),
                          t)
          @unroll for s = 1:nhyperviscstate
            Qhypervisc_grad[ijk, s, e] = l_Qhypervisc[s]
          end
        end
      end
    end
    @synchronize
  end
end

function facehyperviscterms!(bl::BalanceLaw, ::Val{dim}, ::Val{polyorder},
                             ::direction,
                             hyperviscnumflux,
                             Qhypervisc_grad, Qhypervisc_div,
                             Q, auxstate,
                             vgeo, sgeo, vmapM, vmapP,
                             elemtobndy, elems, t) where {dim, polyorder, direction}
  N = polyorder
  FT = eltype(Qhypervisc_grad)
  nstate = num_state(bl,FT)
  ngradlapstate = num_gradient_laplacian(bl,FT)
  nhyperviscstate = num_hyperdiffusive(bl,FT)
  nauxstate = num_aux(bl,FT)
  ngradtransformstate = nstate

  if dim == 1
    Np = (N+1)
    Nfp = 1
    nface = 2
  elseif dim == 2
    Np = (N+1) * (N+1)
    Nfp = (N+1)
    nface = 4
  elseif dim == 3
    Np = (N+1) * (N+1) * (N+1)
    Nfp = (N+1) * (N+1)
    nface = 6
  end

  faces = 1:nface
  if direction == VerticalDirection
    faces = nface-1:nface
  elseif direction == HorizontalDirection
    faces = 1:nface-2
  end

  Nqk = dim == 2 ? 1 : N+1

  l_lapM = MArray{Tuple{ngradlapstate}, FT}(undef)
  l_lapP = MArray{Tuple{ngradlapstate}, FT}(undef)
  l_Qhypervisc = MArray{Tuple{nhyperviscstate}, FT}(undef)
  
  l_QM = MArray{Tuple{ngradtransformstate}, FT}(undef)
  l_auxM = MArray{Tuple{nauxstate}, FT}(undef)

  l_QP = MArray{Tuple{ngradtransformstate}, FT}(undef)
  l_auxP = MArray{Tuple{nauxstate}, FT}(undef)

  @inbounds @loop for e in (elems; blockIdx().x)
    for f in faces
      @loop for n in (1:Nfp; threadIdx().x)
        nM = SVector(sgeo[_n1, n, f, e], sgeo[_n2, n, f, e], sgeo[_n3, n, f, e])
        sM, vMI = sgeo[_sM, n, f, e], sgeo[_vMI, n, f, e]
        idM, idP = vmapM[n, f, e], vmapP[n, f, e]

        eM, eP = e, ((idP - 1) ÷ Np) + 1
        vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

        # Load minus side data
        @unroll for s = 1:ngradtransformstate
          l_QM[s] = Q[vidM, s, eM]
        end
        
        @unroll for s = 1:nauxstate
          l_auxM[s] = auxstate[vidM, s, eM]
        end

        @unroll for s = 1:ngradlapstate
          l_lapM[s] = Qhypervisc_div[vidM, s, eM]
        end

        # Load plus side data
        @unroll for s = 1:ngradtransformstate
          l_QP[s] = Q[vidP, s, eP]
        end
        
        @unroll for s = 1:nauxstate
          l_auxP[s] = auxstate[vidP, s, eP]
        end

        @unroll for s = 1:ngradlapstate
          l_lapP[s] = Qhypervisc_div[vidP, s, eP]
        end

        bctype = elemtobndy[f, e]
        if bctype == 0
          numerical_flux_hyperdiffusive!(hyperviscnumflux, bl, l_Qhypervisc,
                                         nM, l_lapM, l_QM, l_auxM, l_lapP, l_QP, l_auxP, t)
        else
          numerical_boundary_flux_hyperdiffusive!(hyperviscnumflux, bl, l_Qhypervisc,
                                                  nM, l_lapM, l_QM, l_auxM, l_lapP, l_QP, l_auxP, bctype, t)
        end
        
        @unroll for s = 1:nhyperviscstate
          Qhypervisc_grad[vidM, s, eM] += vMI * sM * l_Qhypervisc[s]
        end
      end
      # Need to wait after even faces to avoid race conditions
      f % 2 == 0 && @synchronize
    end
  end
  nothing
end
