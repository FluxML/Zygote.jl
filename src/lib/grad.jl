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
