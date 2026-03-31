@testitem "lib" begin

using ChainRulesTestUtils
using LinearAlgebra: Diagonal, Hermitian, LowerTriangular, UpperTriangular, Symmetric
using LinearAlgebra: UnitLowerTriangular, UnitUpperTriangular
using SparseArrays: sparse, SparseMatrixCSC
using Zygote: ZygoteRuleConfig, _pullback, _reverse

include("lib/number.jl")
include("lib/lib.jl")
include("lib/base.jl")
include("lib/array.jl")

end
