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

@adjoint function meanpool(x, k; kw...)
  y = meanpool(x, k; kw...)
  y, Δ -> (NNlib.∇meanpool(Δ, y, x, k; kw...), nothing)
end
