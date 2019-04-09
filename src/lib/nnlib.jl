using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, maxpool, meanpool, σ

@adjoint function σ(x::Real)
    y = σ(x)
    return y, Δ -> (Δ * y * (1 - y),)
end

@adjoint softmax(xs) = softmax(xs), Δ -> (∇softmax(Δ, xs),)

@adjoint logsoftmax(xs) = logsoftmax(xs), Δ -> (∇logsoftmax(Δ, xs),)

@adjoint NNlib.DenseConvDims(args...; kwargs...) = NNlib.DenseConvDims(args...; kwargs...), _ -> nothing
@adjoint NNlib.DepthwiseConvDims(args...; kwargs...) = NNlib.DepthwiseConvDims(args...; kwargs...), _ -> nothing
@adjoint NNlib.PoolDims(args...; kwargs...) = NNlib.PoolDims(args...; kwargs...), _ -> nothing

@adjoint conv(x, w, cdims; kw...) =
  conv(x, w, cdims; kw...),
    Δ -> begin
       return (
           NNlib.∇conv_data(Δ, w, cdims; kw...),
           NNlib.∇conv_filter(x, Δ, cdims; kw...),
           nothing,
       )
   end

@adjoint ∇conv_data(x, w, cdims; kw...) =
  ∇conv_data(x, w, cdims; kw...),
    Δ -> begin
       return (
           NNlib.conv(Δ, w, cdims; kw...),
           NNlib.∇conv_filter(Δ, x, cdims; kw...),
           nothing,
       )
   end

@adjoint function maxpool(x, pdims; kw...)
  y = maxpool(x, pdims; kw...)
  y, Δ -> (NNlib.∇maxpool(Δ, y, x, pdims; kw...), nothing)
end

@adjoint function meanpool(x, pdims; kw...)
  y = meanpool(x, pdims; kw...)
  y, Δ -> (NNlib.∇meanpool(Δ, y, x, pdims; kw...), nothing)
end
