using MacroTools
using MacroTools: combinedef

named(arg) = isexpr(arg, :(::)) && length(arg.args) == 1 ? :($(gensym())::$(arg.args[1])) : arg

typeless(x) = MacroTools.prewalk(x -> isexpr(x, :(::)) ? x.args[1] : x, x)

tupleornothing(f, ::Nothing) = nothing
tupleornothing(f, x::Tuple) = f(x)
tupleornothing(f, x) = error("Gradient $x should be a tuple")

function adjoint end

function gradm(ex, mut = false)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {Ts__} = body_)) || error("Need a function definition")
  kw = length(args) > 1 && isexpr(args[1], :parameters) ? esc(popfirst!(args)) : nothing
  isclosure = isexpr(name, :(::)) && length(name.args) > 1
  f, T = isexpr(name, :(::)) ?
    (length(name.args) == 1 ? (esc(gensym()), esc(name.args[1])) : esc.(name.args)) :
    (esc(gensym()), :(Core.Typeof($(esc(name)))))
  kT = :(Core.kwftype($T))
  Ts == nothing && (Ts = [])
  args = esc.(named.(args))
  argnames = typeless.(args)
  Ts = esc.(Ts)
  cx = :($(esc(:__context__))::Context)
  fargs = kw == nothing ? [cx, :($f::$T), args...] : [kw, cx, :($f::$T), args...]
  quote
    @inline Zygote.adjoint($(fargs...)) where $(Ts...) = $(esc(body))
    @inline function Zygote._forward($cx, $f::$T, $(args...)) where $(Ts...)
      y, _back = adjoint(__context__, $f, $(argnames...))
      $(mut ? nothing : :(back(::Nothing) = nothing))
      back(Δ) = tupleornothing($(isclosure ? :identity : :(x -> (nothing, x...))),
                               _back(Δ))
      return y, back
    end
    @inline function Zygote._forward($cx, ::$kT, kw, $f::$T, $(args...)) where $(Ts...)
      y, _back = adjoint(__context__, $f, $(argnames...); kw...)
      $(mut ? nothing : :(back(::Nothing) = nothing))
      back(Δ) = tupleornothing($(isclosure ? :(x -> (nothing, nothing, x...)) :
                                             :(x -> (nothing, nothing, nothing, x...))),
                               _back(Δ))
      return y, back
    end
    nothing
  end
end

macro adjoint(ex)
  gradm(ex)
end

macro adjoint!(ex)
  gradm(ex, true)
end

macro nograd(ex)
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = :(;)
  for f in ex.args
    back = MacroTools.@q _ -> ($__source__; nothing)
    push!(blk.args, :(@inline Zygote._forward(::Context, ::Core.Typeof($(esc(f))), args...) = $(esc(f))(args...), $back))
  end
  return blk
end
