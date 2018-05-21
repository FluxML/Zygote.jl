using Zygote, Base.Test
using Zygote: gradient

@testset "Zygote" begin

include("features.jl")
include("gradcheck.jl")

end
