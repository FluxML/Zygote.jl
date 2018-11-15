using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, maxpool, meanpool

@adjoint softmax(xs) = softmax(xs), Δ -> (∇softmax(Δ, xs),)

@adjoint logsoftmax(xs) = logsoftmax(xs), Δ -> (∇logsoftmax(Δ, xs),)

@adjoint conv(x, w; kw...) =
  conv(x, w; kw...),
    Δ ->
      (NNlib.∇conv_data(Δ, x, w; kw...),
       NNlib.∇conv_filter(Δ, x, w; kw...))

@adjoint function maxpool(x, k; kw...)
  y = maxpool(x, k; kw...)
  y, Δ -> (NNlib.∇maxpool(Δ, y, x, k; kw...), nothing)
end

@adjoint function meanpool(x, k; kw...)
  y = meanpool(x, k; kw...)
  y, Δ -> (NNlib.∇meanpool(Δ, y, x, k; kw...), nothing)
end
