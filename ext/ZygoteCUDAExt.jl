module ZygoteCUDAExt

using CUDA: CuArray
using CUDA.CUSPARSE: CuSparseMatrixCSC
using ChainRulesCore: ChainRulesCore, ProjectTo, project_type

# Projection onto the sparsity pattern of a `CuSparseMatrixCSC`, mirroring
# ChainRulesCore's `ProjectTo(::SparseMatrixCSC)` (which keeps gradients of a
# sparse primal sparse) but doing the gather on the GPU so it works without
# scalar indexing. Without this, gradients w.r.t. a `CuSparseMatrixCSC` come
# back as a dense `CuArray`, breaking `update!` and disagreeing with the CPU
# sparse path. See FluxML/Zygote.jl#1313.
function ChainRulesCore.ProjectTo(x::CuSparseMatrixCSC{T}) where {T<:Number}
    m, n = size(x)
    Ti = eltype(x.rowVal)
    # Column index of each stored entry, expanded from `colPtr` on the host
    # (length `n+1`, cheap) and moved to the device once.
    colptr = Array(x.colPtr)
    col = Vector{Ti}(undef, length(x.rowVal))
    @inbounds for c in 1:n, k in colptr[c]:(colptr[c+1]-1)
        col[k] = c
    end
    # Column-major linear index of each stored entry, used to gather from a
    # dense cotangent without scalar indexing.
    lin = x.rowVal .+ Ti(m) .* (CuArray(col) .- one(Ti))
    return ProjectTo{CuSparseMatrixCSC}(;
        element = ProjectTo(zero(T)),
        dims = (m, n),
        colPtr = x.colPtr,
        rowVal = x.rowVal,
        lin = lin,
    )
end

function (project::ProjectTo{CuSparseMatrixCSC})(dx::AbstractArray)
    m, n = project.dims
    size(dx) == (m, n) || throw(DimensionMismatch(
        "variable with size(x) == $(project.dims) cannot have a gradient with size(dx) == $(size(dx))"))
    nzval = vec(dx)[project.lin]
    Tv = project_type(project.element)
    nzval = eltype(nzval) === Tv ? nzval : Tv.(nzval)
    return CuSparseMatrixCSC(project.colPtr, project.rowVal, nzval, project.dims)
end

# Cotangent already sparse: densify (small, no scalar indexing) and reuse the
# gather above so the result has the *primal's* pattern, not the cotangent's.
(project::ProjectTo{CuSparseMatrixCSC})(dx::CuSparseMatrixCSC) = project(CuArray(dx))

end # module
