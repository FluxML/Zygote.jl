using MacroTools: @q

macro nograd(ex)
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = @q begin end
  for f in ex.args
    back = MacroTools.@q _ -> ($__source__; nothing)
    push!(blk.args, :(@inline Zygote._pullback(::Context, ::Core.Typeof($(esc(f))), args...) = $(esc(f))(args...), $back))
  end
  return blk
end

macro which(ex)
  @capture(ex, f_(args__)) || error("Zygote.@which f(args...)")
  :(InteractiveUtils.@which adjoint(Context(), $(esc(f)), $(esc.(args)...)))
end

"""

    checkpointed(f, xs...)

Use gradient checkpointing on the call `f(xs...)`. This means that
`checkpointed(f, xs...) === f(xs...)`, but when computing the derivative
intermediate results from the forward pass of `f` will not be stored. Instead the forward
pass will be repeated, when computing the derivative.
This saves memory at the cost of increasing exectution time.

!!! warning
    If `f` is not a pure function, `checkpointed` will likely give wrong results.
"""
checkpointed(f, xs...) = f(xs...)

function Zygote._pullback(ctx::Zygote.AContext, ::typeof(checkpointed), f, xs...)
    y = f(xs...)
    function pullback_checkpointed(Δy)
        y, pb = Zygote._pullback(ctx, f, xs...)
        return (nothing, pb(Δy)...)
    end
    return y, pullback_checkpointed
end
