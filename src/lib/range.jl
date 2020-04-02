@adjoint StepRange{Int64, Int64}(a::Int64, b::Int64) = a:b, Δ -> (nothing, nothing)
@adjoint StepRange{Int64, Int64}(a::Int64, b::Int64, c::Int64) = a:b:c, Δ -> (nothing, nothing, nothing)
