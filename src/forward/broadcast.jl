using Base.Broadcast: AbstractArrayStyle, broadcasted

@tangent Broadcast.preprocess(dest, bc) =
  Broadcast.preprocess(dest, bc), (ddest, dbc) -> dbc

@tangent broadcasted(::typeof(identity), x::Numeric) = x, (_, ẋ) -> ẋ

@tangent broadcasted(::typeof(+), xs::Numeric...) =
  broadcast(+, xs...), (_, ẋs...) -> broadcast(+, ẋs...)

@tangent function broadcasted(::typeof(tanh), x::Numeric)
  y = tanh.(x)
  y, (_, ẋ) -> ẋ .* (1 .- y.^2)
end

@tangent function broadcasted(::AbstractArrayStyle, f, args...)
  error("Generic broadcast of $f not supported yet")
end
