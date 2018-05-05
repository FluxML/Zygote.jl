# TODO: DiffRules
∇(::typeof(sin), x) = (sin(x), Δ -> (cos(x)*Δ,))
∇(::typeof(cos), x) = (cos(x), Δ -> (-sin(x)*Δ,))

∇(::typeof(+), a, b) = (a+b, Δ -> (Δ, Δ))
∇(::typeof(*), a, b) = (a*b, Δ -> (Δ*b', a'*Δ))
