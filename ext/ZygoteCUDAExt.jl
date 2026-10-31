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
    return ProjectTo{CuSparseMatrixCSC}(;
        element = ProjectTo(zero(T)),
        dims = size(x),
        colPtr = x.colPtr,
        rowVal = x.rowVal,
    )
end

# Column-major linear index of the `k`-th stored entry (in row `r`), with its
# column found by binary search in `colPtr`. Evaluated on the device when the
# projection is applied rather than when it is built, since the `ProjectTo`s of
# most rrule calls are never used. `Int` arithmetic so that `m*n` can't overflow `Int32`.
_linear_index(colPtr, r, k, m) = Int(r) + m * (searchsortedlast(colPtr, k) - 1)

function (project::ProjectTo{CuSparseMatrixCSC})(dx::AbstractArray)
    m, n = project.dims
    size(dx) == (m, n) || throw(DimensionMismatch(
        "variable with size(x) == $(project.dims) cannot have a gradient with size(dx) == $(size(dx))"))
    lin = _linear_index.(Ref(project.colPtr), project.rowVal, eachindex(project.rowVal), m)
    nzval = vec(dx)[lin]
    Tv = project_type(project.element)
    nzval = eltype(nzval) === Tv ? nzval : Tv.(nzval)
    return CuSparseMatrixCSC(project.colPtr, project.rowVal, nzval, project.dims)
end

# Cotangent already sparse: densify (small, no scalar indexing) and reuse the
# gather above so the result has the *primal's* pattern, not the cotangent's.
(project::ProjectTo{CuSparseMatrixCSC})(dx::CuSparseMatrixCSC) = project(CuArray(dx))

end # module
