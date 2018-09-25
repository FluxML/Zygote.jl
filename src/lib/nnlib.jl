using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, maxpool, meanpool

@grad softmax(xs) = softmax(xs), Δ -> (∇softmax(Δ, xs),)

@grad logsoftmax(xs) = logsoftmax(xs), Δ -> (∇logsoftmax(Δ, xs),)

@grad conv(x, w; kw...) =
  conv(x, w; kw...),
    Δ ->
      (NNlib.∇conv_data(Δ, x, w; kw...),
       NNlib.∇conv_filter(Δ, x, w; kw...))

@grad function maxpool(x, k; kw...)
  let y = maxpool(x, k; kw...)
    y, Δ -> (NNlib.∇maxpool(Δ, y, x, k; kw...), nothing)
  end
end

@grad function meanpool(x, k; kw...)
  let y = meanpool(x, k; kw...)
    y, Δ -> (NNlib.∇meanpool(Δ, y, x, k; kw...), nothing)
  end
end
