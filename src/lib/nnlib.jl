using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, maxpool, meanpool, _conv

@adjoint softmax(xs) = softmax(xs), Δ -> (∇softmax(Δ, xs),)

@adjoint logsoftmax(xs) = logsoftmax(xs), Δ -> (∇logsoftmax(Δ, xs),)

@adjoint (::_conv{pad, stride, dilation})(x, w) where {pad, stride, dilation} =
  _conv{pad, stride, dilation}()(x, w),
    Δ ->
      (NNlib._∇conv_data{pad, stride, dilation, 0}()(Δ, x, w),
       NNlib._∇conv_filter{pad, stride, dilation, 0}()(Δ, x, w))

@adjoint function maxpool(x, k; kw...)
  y = maxpool(x, k; kw...)
  y, Δ -> (NNlib.∇maxpool(Δ, y, x, k; kw...), nothing)
end

struct ∇meanpool{k, pad, stride, T, S}
  x::T
  y::S
end
∇meanpool{k, pad, stride}(x::T, y::S) where {k, pad, stride, T, S} = ∇meanpool{k, pad, stride, T, S}(x, y)
(mp::∇meanpool{k, pad, stride})(Δ) where {k, pad, stride} = (NNlib.∇meanpool(Δ, mp.y, mp.x, k; pad=pad, stride=stride), nothing)

@adjoint function meanpool(x, k; pad = map(_->0,k), stride = k)
  let y = meanpool(x, k; pad=pad, stride=stride)
    y, ∇meanpool{k, pad, stride}(x, y)
  end
end
