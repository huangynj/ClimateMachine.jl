
export BatchedGeneralizedMinimalResidual

mutable struct BatchedGeneralizedMinimalResidual{
    MP1,
    Nbatch,
    I,
    T,
    AT,
    FRS,
    FPR,
    BRS,
    BPR
} <: AbstractIterativeSystemSolver

    "global Krylov basis at present step"
    krylov_basis::AT
    "global Krylov basis at previous step"
    krylov_basis_m::AT
    "global batched Krylov basis"
    batched_krylov_basis::NTuple{MP1, Matrix{T}}
    "Hessenberg matrix in each column"
    H::NTuple{Nbatch, Matrix{T}}
    "rhs of the least squares problem in each column"
    g0::NTuple{Nbatch, Vector{T}}
    "solution of the least squares problem in each column"
    y::NTuple{Nbatch, Vector{T}}
    "The GMRES iterate in each batched column"
    sol::Matrix{T}
    rtol::T
    atol::T
    "Maximum number of GMRES iterations (global across all columns)"
    max_iter::I
    "total number of batched columns"
    batch_size::I
    "iterations to reach convergence in each column"
    iterconv::Vector{I}
    "residual norm in each column"
    resnorms::Vector{T}
    forward_reshape::FRS
    forward_permute::FPR
    backward_reshape::BRS
    backward_permute::BPR

    function BatchedGeneralizedMinimalResidual(
        Q::AT,
        dofperbatch,
        Nbatch;
        M = min(20, length(Q)),
        rtol = √eps(eltype(AT)),
        atol = eps(eltype(AT)),
        forward_reshape = size(Q),
        forward_permute = Tuple(1:length(size(Q))),
    ) where {AT}

        krylov_basis = similar(Q)
        krylov_basis_m = similar(Q)
        # TODO: Need to check this. These are work vectors to
        # 'batch' the krylov basis vectors
        # batched_krylov_basis = -zeros(eltype(AT), (M + 1, Nbatch, dofperbatch))
        batched_krylov_basis = ntuple(i -> zeros(eltype(AT), dofperbatch, Nbatch),  M + 1)

        # H = -zeros(eltype(AT), (Nbatch, M + 1, M))
        H = ntuple(i -> zeros(eltype(AT), M + 1, M), Nbatch)
        # g0 = -zeros(eltype(AT), (Nbatch, M + 1))
        g0 = ntuple(i -> zeros(eltype(AT), M + 1), Nbatch)

        y = ntuple(i -> zeros(eltype(AT), M + 1), Nbatch)
    
        sol = zeros(eltype(AT), dofperbatch, Nbatch)

        @assert dofperbatch * Nbatch == length(Q)

        iterconv = fill(-1, Nbatch)
        resnorms = -zeros(eltype(AT), Nbatch)

        FRS = typeof(forward_reshape)
        FPR = typeof(forward_permute)
        # define the back permutation and reshape
        backward_permute = Tuple(sortperm([forward_permute...]))
        tmp_reshape_tuple_b = [forward_reshape...]
        permute!(tmp_reshape_tuple_b, [forward_permute...])
        backward_reshape = Tuple(tmp_reshape_tuple_b)
        BRS = typeof(backward_reshape)
        BPR = typeof(backward_permute)

        @info "Initialize bgmres : "
        @show dofperbatch, Nbatch, M
        @show forward_reshape, forward_permute
        @show backward_reshape, backward_permute

        new{M + 1, Nbatch, typeof(Nbatch), eltype(Q), AT,  FRS, FPR, BRS, BPR}(
            krylov_basis,
            krylov_basis_m,
            batched_krylov_basis,
            H,
            g0,
            y,
            sol,
            rtol,
            atol,
            M,
            Nbatch,
            iterconv,
            resnorms,
            forward_reshape,
            forward_permute,
            backward_reshape,
            backward_permute,
        )
    end
end

"""
    BatchedGeneralizedMinimalResidual(
        dg::DGModel,
        Q::MPIStateArray;
        atol = sqrt(eps(eltype(Q))),
        rtol = sqrt(eps(eltype(Q))),
        max_iteration = nothing,
    )

# Description
Specialized constructor for `BatchedGeneralizedMinimalResidual` struct, using
a `DGModel` to infer state-information and determine appropriate reshaping
and permutations.

# Arguments
- `dg`: (DGModel) A `DGModel` containing all relevant grid and topology
        information.
- `Q` : (MPIStateArray) An `MPIStateArray` containing field information.

# Keyword Arguments
- `atol`: (float) absolute tolerance. `DEFAULT = sqrt(eps(eltype(Q)))`
- `rtol`: (float) relative tolerance. `DEFAULT = sqrt(eps(eltype(Q)))`
- `max_iteration` : (Int).    Maximal dimension of each (batched)
                              Krylov subspace. DEFAULT = nothing
# Return
instance of `BatchedGeneralizedMinimalResidual` struct
"""
function BatchedGeneralizedMinimalResidual(
    dg::DGModel,
    Q::MPIStateArray;
    atol = sqrt(eps(eltype(Q))),
    rtol = sqrt(eps(eltype(Q))),
    max_iteration = nothing,
)

    # Need to determine array type for storage vectors
    if isa(Q.data, Array)
        ArrayType = Array
    else
        ArrayType = CuArray
    end

    grid = dg.grid
    topology = grid.topology
    dim = dimensionality(grid)

    # Number of Gauss-Lobatto quadrature points in 1D
    Nq = polynomialorder(grid) + 1

    # Assumes same number of quadrature points in all spatial directions
    Np = Tuple([Nq for i in 1:dim])

    # Number of states and elements (in vertical and horizontal directions)
    num_states = size(Q)[2]
    nelem = length(topology.elems)
    nvertelem = topology.stacksize
    nhorzelem = div(nelem, nvertelem)

    # Definition of a "column" here is a vertical stack of degrees
    # of freedom. For example, consider a mesh consisting of a single
    # linear element:
    #    o----------o
    #    |\ d1   d2 |\
    #    | \        | \
    #    |  \ d3    d4 \
    #    |   o----------o
    #    o--d5---d6-o   |
    #     \  |       \  |
    #      \ |        \ |
    #       \|d7    d8 \|
    #        o----------o
    # There are 4 total 1-D columns, each containing two
    # degrees of freedom. In general, a mesh of stacked elements will
    # have `Nq^2 * nhorzelem` total 1-D columns.
    # A single 1-D column has `Nq * nvertelem * num_states`
    # degrees of freedom.
    #
    # nql = length(Np)
    # indices:      (1...nql, nql + 1 , nql + 2, nql + 3)

    # for 3d case, this is [ni, nj, nk, num_states, nvertelem, nhorzelem]
    # here ni, nj, nk are number of Gauss quadrature points in each element in x-y-z directions
    # Q = reshape(Q, reshaping_tup), leads to the column-wise fashion Q
    reshaping_tup = (Np..., num_states, nvertelem, nhorzelem)

    m = Nq * nvertelem * num_states
    n = (Nq^(dim - 1)) * nhorzelem

    if max_iteration === nothing
        max_iteration = m
    end

    # permute [ni, nj, nk, num_states, nvertelem, nhorzelem] 
    # to      [nvertelem, nk, num_states, ni, nj, nhorzelem]
    permute_size = length(reshaping_tup)
    permute_tuple_f = (dim + 2, dim, dim + 1, (1:dim-1)..., permute_size)

    return NewBatchedGeneralizedMinimalResidual(
        Q,
        n,
        m;
        M = max_iteration,
        atol = atol,
        rtol = rtol,
        forward_reshape = reshaping_tup,
        forward_permute = permute_tuple_f,
    )
end

@inline function convert_structure!(
    x,
    y,
    reshape_tuple,
    permute_tuple,
)
    alias_y = reshape(y, reshape_tuple)
    permute_y = permutedims(alias_y, permute_tuple)
    x[:] .= permute_y[:]
    nothing
end
@inline convert_structure!(x, y::MPIStateArray, reshape_tuple, permute_tuple) =
    convert_structure!(x, y.data, reshape_tuple, permute_tuple)
@inline convert_structure!(x::MPIStateArray, y, reshape_tuple, permute_tuple) =
    convert_structure!(x.data, y, reshape_tuple, permute_tuple)

function initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    args...,
)
    g0 = solver.g0
    krylov_basis = solver.krylov_basis
    rtol, atol = solver.rtol, solver.atol

    iterconv = solver.iterconv
    batched_krylov_basis = solver.batched_krylov_basis
    forward_reshape = solver.forward_reshape
    forward_permute = solver.forward_permute
    resnorms = solver.resnorms

    # Get device and groupsize information
    device = array_device(Q)
    if isa(device, CPU)
        groupsize = Threads.nthreads()
    else # isa(device, CUDADevice)
        groupsize = 256
    end

    @assert size(Q) == size(krylov_basis)

    # FIXME: Can we make linearoperator! batch-able?
    # store the initial residual in krylov_basis[1] = r0/|r0|
    linearoperator!(krylov_basis, Q, args...)
    @. krylov_basis = Qrhs - krylov_basis

    convert_structure!( 
        batched_krylov_basis[1],
        krylov_basis,
        forward_reshape,
        forward_permute,
    )

    event = Event(device)
    event = batched_initialize!(device, groupsize)(
        resnorms,
        iterconv,
        g0,
        batched_krylov_basis[1];
        ndrange = solver.batch_size,
        dependencies = (event,),
    )
    wait(device, event)

    residual_norm = maximum(resnorms)
    threshold = rtol * residual_norm
    converged = false
    if threshold < atol
        converged = true
    end

    @info "Calling initialize: converged = $converged, residual (max) = $residual_norm"

    converged, max(threshold, atol)
end

"""
update batched_krylov_basis r0/|r0|
update rhs g0 = βe1, where β = |r0|
update resnorms = |r0|
update iterconv = 0
"""
@kernel function batched_initialize!(resnorms, iterconv, g0, batched_krylov_basis)
    cidx = @index(Global)

    FT = eltype(batched_krylov_basis)
    fill!(g0[cidx], FT(0.0))

    local_residual_norm = norm(batched_krylov_basis[:, cidx], false)
    g0[cidx][1] = local_residual_norm
    @. batched_krylov_basis[:, cidx] /= local_residual_norm
    resnorms[cidx] = local_residual_norm
    iterconv[cidx] = 0

    nothing
end

function doiteration!(
    linearoperator!,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    threshold,
    args...,
)
    FT = eltype(Q)
    krylov_basis = solver.krylov_basis
    krylov_basis_m = solver.krylov_basis_m
    Hs = solver.H
    g0s = solver.g0
    ys = solver.y
    sols = solver.sol
    iterconv = solver.iterconv
    batched_krylov_basis = solver.batched_krylov_basis
    forward_reshape = solver.forward_reshape
    forward_permute = solver.forward_permute
    backward_reshape = solver.backward_reshape
    backward_permute = solver.backward_permute
    resnorms = solver.resnorms

    # Get device and groupsize information
    device = array_device(Q)
    if isa(device, CPU)
        groupsize = Threads.nthreads()
    else # isa(device, CUDADevice)
        groupsize = 256
    end

    converged = false
    residual_norm = typemax(FT)
    Ωs = ntuple(i->LinearAlgebra.Rotation{FT}([]), solver.batch_size)
    j = 1
    @info "Current threshold: $threshold"
    for outer j in 1:solver.max_iter
        # FIXME: To make this a truly batched method, we need to be able
        # to make operator application batch-able.
        # Global operator matvec

        convert_structure!( 
            krylov_basis_m,
            batched_krylov_basis[j],
            backward_reshape,
            backward_permute,
        )

        linearoperator!(krylov_basis, krylov_basis_m, args...)

        # Now that we have a global Krylov vector, we reshape and batch
        # the Arnoldi iterations
        convert_structure!( 
            batched_krylov_basis[j + 1],
            krylov_basis,
            forward_reshape,
            forward_permute,
        )

        event = Event(device)
        event = batched_arnoldi_process!(device, groupsize)(
            resnorms,
            g0s,
            Hs,
            Ωs,
            batched_krylov_basis,
            j;
            ndrange = solver.batch_size,
            dependencies = (event,),
        )
        wait(device, event)

        # Converge when all columns are converged
        residual_norm = maximum(resnorms)
        @info "Max residual at iteration $j: $residual_norm"
        if residual_norm < threshold
            converged = true
            break
        end
    end

       ## compose the solution todo use batched_krylov_basis
    #todo put in the initilization
    convert_structure!( 
        sols,
        Q,
        forward_reshape,
        forward_permute,
    )

    # solve the triangular system and construct the
    # gmres iterate in each column
    event = Event(device)
    event = construct_batched_gmres_iterate!(device, groupsize)(
        batched_krylov_basis,
        Hs,
        g0s,
        ys,
        sols,
        j;
        ndrange = solver.batch_size,
        dependencies = (event,),
    )
    wait(device, event)

    convert_structure!( 
        Q,
        sols,
        backward_reshape,
        backward_permute,
    )

    @info "after bgmres iterations, converged, j, residual_norm : ",  converged, j, residual_norm

    # if not converged, then restart
    converged || initialize!(linearoperator!, Q, Qrhs, solver, args...)

    (converged, j, residual_norm)
end

@kernel function batched_arnoldi_process!(resnorms, g0s, Hs, Ωs, batched_krylov_basis, j)
    cidx = @index(Global)
    g0 = g0s[cidx]
    H = Hs[cidx]
    Ω = Ωs[cidx]

    for i in 1:j
        H[i, j] = dot(batched_krylov_basis[j + 1][:, cidx], batched_krylov_basis[i][:, cidx], false)
        @. batched_krylov_basis[j + 1][:, cidx] -= H[i, j] * batched_krylov_basis[i][:, cidx]
    end
    H[j + 1, j] = norm(batched_krylov_basis[j + 1][:, cidx], false)
    batched_krylov_basis[j + 1][:, cidx] ./= H[j + 1, j]

    # apply the previous Givens rotations to the new column of H
    @views H[1:j, j:j] .= Ω * H[1:j, j:j]

    # compute a new Givens rotation to zero out H[j + 1, j]
    G, _ = givens(H, j, j + 1, j)

    # apply the new rotation to H and the rhs
    H .= G * H
    g0 .= G * g0

    # Compose the new rotation with the others
    Ω = lmul!(G, Ω)
    resnorms[cidx] = abs(g0[j + 1])

    nothing
end

@kernel function construct_batched_gmres_iterate!(batched_krylov_basis, Hs, g0s, ys, sols, j)
    cidx = @index(Global)
    g0 = g0s[cidx]
    H = Hs[cidx]
    y = ys[cidx]
    y[1:j] .= UpperTriangular(H[1:j, 1:j]) \ g0[1:j]

    for i in 1:j
        sols[:, cidx] .+= y[i] * batched_krylov_basis[i][:, cidx]
    end
    nothing
end
