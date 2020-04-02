using MacroTools: @q

"""
    @nograd(f...)

The output of the argument functions will be considered as constant
and will not contribute gradients.
"""
macro nograd(ex)
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = @q begin end
  for f in ex.args
    back = MacroTools.@q _ -> ($__source__; nothing)
    push!(blk.args, :(@inline Zygote._pullback(::Context, ::Core.Typeof($(esc(f))), args...) = $(esc(f))(args...), $back))
  end
  return blk
end

"""
    nograd(x)

Treats `x` as a constant. For example:

```julia-repl
julia> f(x) = sum(nograd(x) + x); gradient(f, randn(5))
([1.0, 1.0, 1.0, 1.0, 1.0],)
```

In the example above, the gradient is an array of ones even though
`f(x)` computes `sum(2x)`, because `nograd(x)` uses the value of `x`
as if it were a constant.
"""
nograd(x) = x
@nograd nograd

macro which(ex)
  @capture(ex, f_(args__)) || error("Zygote.@which f(args...)")
  :(InteractiveUtils.@which adjoint(Context(), $(esc(f)), $(esc.(args)...)))
end
