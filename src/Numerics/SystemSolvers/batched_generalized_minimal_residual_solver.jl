
export BatchedGeneralizedMinimalResidual

"""
    BatchedGeneralizedMinimalResidual(
        Q,
        dofperbatch,
        Nbatch;
        M = min(20, length(Q)),
        rtol = √eps(eltype(AT)),
        atol = eps(eltype(AT)),
        forward_reshape = size(Q),
        forward_permute = Tuple(1:length(size(Q))),
    )

# BGMRES
This is an object for solving batched linear systems using the GMRES algorithm.
The constructor parameter `M` is the number of steps after which the algorithm
is restarted (if it has not converged), `Q` is a reference state used only
to allocate the solver internal state, `dofperbatch` is the size of each batched
system (assumed to be the same throughout), `Nbatch` is the total number
of independent linear systems, and `rtol` specifies the convergence
criterion based on the relative residual norm (max across all batched systems).
The argument `forward_reshape` is a tuple of integers denoting the reshaping
(if required) of the solution vectors for batching the Arnoldi routines.
The argument `forward_permute` describes precisely which indices of the
array `Q` to permute. This object is intended to be passed to
the [`linearsolve!`](@ref) command.

This uses a batched-version of the restarted Generalized Minimal Residual method
of Saad and Schultz (1986).

# Note
Eventually, we'll want to do something like this:

    i = @index(Global)
    linearoperator!(Q[:, :, :, i], args...)

This will help stop the need for constantly
reshaping the work arrays. It would also potentially
save us some memory.
"""
mutable struct BatchedGeneralizedMinimalResidual{
    MP1,
    Nbatch,
    I,
    T,
    AT,
    FRS,
    FPR,
    BRS,
    BPR,
} <: AbstractIterativeSystemSolver

    "global Krylov basis at present step"
    krylov_basis::AT
    "global Krylov basis at previous step"
    krylov_basis_prev::AT
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
    "residual norm in each column"
    resnorms::Vector{T}
    "residual norm in each column"
    resnorms0::Vector{T}
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
        # Since the application of linearoperator! is not currently batch-able,
        # we need two temporary work vectors to hold the current and previous Krylov
        # basis vector
        krylov_basis = similar(Q)
        krylov_basis_prev = similar(Q)

        # Get ArrayType information
        if isa(array_device(Q), CPU)
            ArrayType = Array
        else
            # Sanity check since we don't support anything else
            @assert isa(array_device(Q), CUDADevice)
            ArrayType = CuArray
        end

        # Create storage for holding the batched Krylov basis
        batched_krylov_basis = ntuple(
            i -> ArrayType(-zeros(eltype(AT), dofperbatch, Nbatch)),
            M + 1,
        )
        # Create storage for doing the batched Arnoldi process
        H = ntuple(i -> ArrayType(-zeros(eltype(AT), M + 1, M)), Nbatch)
        g0 = ntuple(i -> ArrayType(-zeros(eltype(AT), M + 1)), Nbatch)
        y = ntuple(i -> ArrayType(-zeros(eltype(AT), M + 1)), Nbatch)
        sol = ArrayType(-zeros(eltype(AT), dofperbatch, Nbatch))
        resnorms = ArrayType(-zeros(eltype(AT), Nbatch))
        resnorms0 = ArrayType(-zeros(eltype(AT), Nbatch))

        @assert dofperbatch * Nbatch == length(Q)

        FRS = typeof(forward_reshape)
        FPR = typeof(forward_permute)

        # define the back permutation and reshape
        backward_permute = Tuple(sortperm([forward_permute...]))
        tmp_reshape_tuple_b = [forward_reshape...]
        permute!(tmp_reshape_tuple_b, [forward_permute...])
        backward_reshape = Tuple(tmp_reshape_tuple_b)
        BRS = typeof(backward_reshape)
        BPR = typeof(backward_permute)

        new{M + 1, Nbatch, typeof(Nbatch), eltype(Q), AT, FRS, FPR, BRS, BPR}(
            krylov_basis,
            krylov_basis_prev,
            batched_krylov_basis,
            H,
            g0,
            y,
            sol,
            rtol,
            atol,
            M,
            Nbatch,
            resnorms,
            resnorms0,
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
"""
function BatchedGeneralizedMinimalResidual(
    dg::DGModel,
    Q::MPIStateArray;
    atol = sqrt(eps(eltype(Q))),
    rtol = sqrt(eps(eltype(Q))),
    max_iteration = nothing,
)
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
    permute_tuple_f = (dim + 2, dim, dim + 1, (1:(dim - 1))..., permute_size)

    return BatchedGeneralizedMinimalResidual(
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

function initialize!(
    linearoperator!,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    args...;
    restart = false,
)
    g0 = solver.g0
    krylov_basis = solver.krylov_basis
    rtol, atol = solver.rtol, solver.atol

    batched_krylov_basis = solver.batched_krylov_basis
    forward_reshape = solver.forward_reshape
    forward_permute = solver.forward_permute
    resnorms = solver.resnorms
    resnorms0 = solver.resnorms0
    # Get device and groupsize information
    device = array_device(Q)
    if isa(device, CPU)
        groupsize = Threads.nthreads()
    else # isa(device, CUDADevice)
        groupsize = 256
    end

    @assert size(Q) == size(krylov_basis)

    # FIXME: Can we make linearoperator! batch-able?
    # store the initial (global) residual in krylov_basis = r0/|r0|
    linearoperator!(krylov_basis, Q, args...)
    krylov_basis .= Qrhs .- krylov_basis

    # Convert into a batched Krylov basis vector
    convert_structure!(
        batched_krylov_basis[1],
        krylov_basis,
        forward_reshape,
        forward_permute,
    )

    # Now we initialize across all columns (solver.batch_size).
    # This function also computes the residual norm in each column
    event = Event(device)
    event = batched_initialize!(device, groupsize)(
        resnorms,
        g0,
        batched_krylov_basis[1];
        ndrange = solver.batch_size,
        dependencies = (event,),
    )
    wait(device, event)

    

    # When restarting, we do not want to overwrite the initial threshold,
    # otherwise we may not get an accurate indication that we have sufficiently
    # reduced the GMRES residual.
    if !restart
        resnorms0 .= resnorms
    end

    converged,  residual_norm = check_convergence(resnorms, resnorms0, atol, rtol)
    # if threshold === nothing
    #     threshold = rtol * residual_norm
    #     if threshold < atol
    #         converged = true
    #     end
    # else
    #     # if restarting, then threshold has
    #     # already been computed and we simply check
    #     # the current residual norm
    #     if residual_norm < threshold
    #         converged = true
    #     end
    # end

    converged, residual_norm
end

function doiteration!(
    linearoperator!,
    Q,
    Qrhs,
    solver::BatchedGeneralizedMinimalResidual,
    args...,
)
    FT = eltype(Q)
    krylov_basis = solver.krylov_basis
    krylov_basis_prev = solver.krylov_basis_prev
    Hs = solver.H
    g0s = solver.g0
    ys = solver.y
    sols = solver.sol
    batched_krylov_basis = solver.batched_krylov_basis
    rtol, atol = solver.rtol, solver.atol
    
    forward_reshape = solver.forward_reshape
    forward_permute = solver.forward_permute
    backward_reshape = solver.backward_reshape
    backward_permute = solver.backward_permute
    resnorms = solver.resnorms
    resnorms0 = solver.resnorms0

    # Get device and groupsize information
    device = array_device(Q)
    if isa(device, CPU)
        groupsize = Threads.nthreads()
    else # isa(device, CUDADevice)
        groupsize = 256
    end

    # Main batched-GMRES iteration cycle
    converged = false
    residual_norm = typemax(FT)
    Ωs = ntuple(i -> LinearAlgebra.Rotation{FT}([]), solver.batch_size)
    j = 1
    for outer j in 1:(solver.max_iter)
        # FIXME: To make this a truly batched method, we need to be able
        # to make operator application batch-able. That way, we don't have
        # to do this back-and-forth reshaping
        convert_structure!(
            krylov_basis_prev,
            batched_krylov_basis[j],
            backward_reshape,
            backward_permute,
        )

        # Global operator application to get new Krylov basis vector
        linearoperator!(krylov_basis, krylov_basis_prev, args...)

        # Now that we have a global Krylov vector, we reshape and batch
        # the Arnoldi iterations across all columns
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

        # Current stopping criteria is based on the maximal column norm
        # TODO: Once we are able to batch the operator application, we
        # should revisit the termination criteria.
        converged, residual_norm = check_convergence(resnorms, resnorms0, atol, rtol)
        if converged; break; end

        # residual_norm = maximum(resnorms)
        # if residual_norm < threshold
        #     converged = true
        #     break
        # end
    end

    # Reshape the solution vector to construct the new GMRES iterate
    convert_structure!(sols, Q, forward_reshape, forward_permute)

    # Solve the triangular system (minimization problem for optimal linear coefficients
    # in the GMRES iterate) and construct the current iterate in each column
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

    # Unwind reshaping and return solution in standard format
    convert_structure!(Q, sols, backward_reshape, backward_permute)

    # if not converged, then restart
    converged || initialize!(
        linearoperator!,
        Q,
        Qrhs,
        solver,
        args...;
        restart = true,
    )

    (converged, j, residual_norm)
end

@kernel function batched_initialize!(resnorms, g0, batched_krylov_basis)
    cidx = @index(Global)
    FT = eltype(batched_krylov_basis)
    fill!(g0[cidx], FT(0.0))

    # Compute the column norm of the initial residual r0
    local_residual_norm = norm(batched_krylov_basis[:, cidx], false)
    # Set g0 = βe1, where β = |r0|
    g0[cidx][1] = local_residual_norm
    # Normalize the batched_krylov_basis by the (local) residual norm
    batched_krylov_basis[:, cidx] ./= local_residual_norm
    # Record initialize residual norm in the column
    resnorms[cidx] = local_residual_norm

    nothing
end

@kernel function batched_arnoldi_process!(
    resnorms,
    g0s,
    Hs,
    Ωs,
    batched_krylov_basis,
    j,
)
    cidx = @index(Global)
    g0 = g0s[cidx]
    H = Hs[cidx]
    Ω = Ωs[cidx]

    # Arnoldi process in the local column `cidx`
    @inbounds for i in 1:j
        H[i, j] = dot(
            batched_krylov_basis[j + 1][:, cidx],
            batched_krylov_basis[i][:, cidx],
            false,
        )
        batched_krylov_basis[j + 1][:, cidx] .-=
            H[i, j] * batched_krylov_basis[i][:, cidx]
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

@kernel function construct_batched_gmres_iterate!(
    batched_krylov_basis,
    Hs,
    g0s,
    ys,
    sols,
    j,
)
    # Solve for the GMRES coefficients (yⱼ) at the `j`-th
    # iteration that minimizes ∥ b - A xⱼ ∥_2, where
    # xⱼ = ∑ᵢ yᵢ Ψᵢ, with Ψᵢ denoting the Krylov basis vectors
    cidx = @index(Global)
    g0 = g0s[cidx]
    H = Hs[cidx]
    y = ys[cidx]
    # TODO: Is this GPU-safe?
    y[1:j] .= UpperTriangular(H[1:j, 1:j]) \ g0[1:j]

    # Having determined yᵢ, we now construct the GMRES solution
    # in each column: xⱼ = ∑ᵢ yᵢ Ψᵢ
    @inbounds for i in 1:j
        sols[:, cidx] .+= y[i] * batched_krylov_basis[i][:, cidx]
    end

    nothing
end

"""
    convert_structure!(
        x,
        y,
        reshape_tuple,
        permute_tuple,
    )

Computes a tensor transpose and stores result in x

# Arguments
- `x`: (array) [OVERWRITTEN]. target destination for storing the y data
- `y`: (array). data that we want to copy
- `reshape_tuple`: (tuple) reshapes y to be like that of x, up to a permutation
- `permute_tuple`: (tuple) permutes the reshaped array into the correct structure
"""
@inline function convert_structure!(x, y, reshape_tuple, permute_tuple)
    alias_y = reshape(y, reshape_tuple)
    permute_y = permutedims(alias_y, permute_tuple)
    x[:] .= permute_y[:]
    nothing
end
@inline convert_structure!(x, y::MPIStateArray, reshape_tuple, permute_tuple) =
    convert_structure!(x, y.data, reshape_tuple, permute_tuple)
@inline convert_structure!(x::MPIStateArray, y, reshape_tuple, permute_tuple) =
    convert_structure!(x.data, y, reshape_tuple, permute_tuple)



function check_convergence(resnorms, resnorms0, atol, rtol)

    # Current stopping criteria is based on the maximal column norm
    residual_norm = maximum(resnorms)
    residual_norm0 = maximum(resnorms0)
    threshold = residual_norm0 * rtol
    converged  = (residual_norm < threshold)
    return converged, residual_norm
end