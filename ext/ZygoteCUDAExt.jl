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

function _check_dims(project::ProjectTo{CuSparseMatrixCSC}, dx)
    size(dx) == project.dims || throw(DimensionMismatch(
        "variable with size(x) == $(project.dims) cannot have a gradient with size(dx) == $(size(dx))"))
end

function _with_pattern(project::ProjectTo{CuSparseMatrixCSC}, nzval)
    Tv = project_type(project.element)
    nzval = eltype(nzval) === Tv ? nzval : Tv.(nzval)
    return CuSparseMatrixCSC(project.colPtr, project.rowVal, nzval, project.dims)
end

function (project::ProjectTo{CuSparseMatrixCSC})(dx::AbstractArray)
    _check_dims(project, dx)
    m = project.dims[1]
    lin = _linear_index.(Ref(project.colPtr), project.rowVal, eachindex(project.rowVal), m)
    return _with_pattern(project, vec(dx)[lin])
end

# Value stored at row `r`, column `c` of a CSC matrix, or zero. Needs the row
# indices of each column to be sorted.
function _stored_value(colPtr, rowVal, nzVal, r, c)
    lo, hi = colPtr[c], colPtr[c+1] - 1
    i = lo - 1 + searchsortedfirst(view(rowVal, lo:hi), r)
    return i <= hi && rowVal[i] == r ? nzVal[i] : zero(eltype(nzVal))
end

# Whether entry `k`, in row `r`, is in row order with entry `k-1` of its column.
_row_ordered(colPtr, rowVal, r, k) = k == colPtr[searchsortedlast(colPtr, k)] || rowVal[k-1] <= r

# Cotangent already sparse: look the primal's stored entries up in it, so that
# the result has the *primal's* pattern, not the cotangent's, without
# materialising the dense matrix.
function (project::ProjectTo{CuSparseMatrixCSC})(dx::CuSparseMatrixCSC)
    _check_dims(project, dx)
    if dx.colPtr === project.colPtr && dx.rowVal === project.rowVal
        return _with_pattern(project, dx.nzVal)
    elseif !all(_row_ordered.(Ref(dx.colPtr), Ref(dx.rowVal), dx.rowVal, eachindex(dx.rowVal)))
        return project(CuArray(dx))
    end
    nzval = _stored_value.(Ref(dx.colPtr), Ref(dx.rowVal), Ref(dx.nzVal), project.rowVal,
                           searchsortedlast.(Ref(project.colPtr), eachindex(project.rowVal)))
    return _with_pattern(project, nzval)
end

end # module
