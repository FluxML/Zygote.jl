@adjoint StepRange{Int64, Int64}(a::Int64, b::Int64) = a:b, Δ -> (nothing, nothing)
@adjoint StepRange{Int64, Int64}(a::Int64, b::Int64, c::Int64) = a:b:c, Δ -> (nothing, nothing, nothing)

# ChainRules marks `(:)` as `@non_differentiable`, which silently drops the gradient
# w.r.t. the endpoints of a *real* range (issue #954). The endpoints carry a genuine
# gradient: for `r = start:step:stop`, `r[i] = start + (i-1)*step`, while `stop` only
# determines the length and is non-differentiable.
@adjoint function (:)(start::Real, stop::Real)
  start:stop, function (Δ)
    Δ === nothing && return (nothing, nothing)
    (sum(Δ), nothing)
  end
end

@adjoint function (:)(start::Real, step::Real, stop::Real)
  start:step:stop, function (Δ)
    Δ === nothing && return (nothing, nothing, nothing)
    dstep = sum(i -> (i - 1) * Δ[i], eachindex(Δ); init = zero(eltype(Δ)) * zero(step))
    (sum(Δ), dstep, nothing)
  end
end
