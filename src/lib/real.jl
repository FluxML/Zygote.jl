grad(x::Real) = zero(x)
grad(x::Integer) = zero(float(x))

# TODO: DiffRules
@grad sin(x) = sin(x), Δ -> (cos(x)*Δ,)
@grad cos(x) = cos(x), Δ -> (-sin(x)*Δ,)

@grad a + b = a+b, Δ -> (Δ, Δ)
@grad a * b = a*b, Δ -> (Δ*b', a'*Δ)
