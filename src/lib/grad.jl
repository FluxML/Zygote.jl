using MacroTools
using MacroTools: combinedef

named(arg) = isexpr(arg, :(::)) && length(arg.args) == 1 ? :($(gensym())::$(arg.args[1])) : arg

typeless(x) = MacroTools.prewalk(x -> isexpr(x, :(::)) ? x.args[1] : x, x)

_gradtuple(::Nothing) = nothing
_gradtuple(x::Tuple) = (nothing, x...)
_gradtuple(x) = error("Gradient $x should be a tuple")

_gradtuple_kw(::Nothing) = nothing
_gradtuple_kw(x::Tuple) = (nothing, nothing, nothing, x...) # kwfunc, kws, func, args...
_gradtuple_kw(x) = error("Gradient $x should be a tuple")

function grad end

function gradm(ex, mut = false)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {Ts__} = body_)) || error("Need a function definition")
  kw = length(args) > 1 && isexpr(args[1], :parameters) ? esc(popfirst!(args)) : nothing
  f, T = isexpr(name, :(::)) ?
    (length(name.args) == 1 ? (esc(gensym()), esc(name.args[1])) : esc.(name.args)) :
    (esc(gensym()), :(typeof($(esc(name)))))
  kT = :(Core.kwftype($T))
  Ts == nothing && (Ts = [])
  args = esc.(named.(args))
  argnames = typeless.(args)
  Ts = esc.(Ts)
  cx = :($(esc(:__context__))::Context)
  fargs = kw == nothing ? [cx, :($f::$T), args...] : [kw, cx, :($f::$T), args...]
  quote
    @inline Zygote.grad($(fargs...)) where $(Ts...) = $(esc(body))
    @inline function Zygote._forward($cx, $f::$T, $(args...)) where $(Ts...)
      y, _back = grad(__context__, $f, $(argnames...))
      $(mut ? nothing : :(back(::Nothing) = nothing))
      back(Δ) = _gradtuple(_back(Δ))
      return y, back
    end
    @inline function Zygote._forward($cx, ::$kT, kw, $f::$T, $(args...)) where $(Ts...)
      y, _back = grad(__context__, $f, $(argnames...); kw...)
      $(mut ? nothing : :(back(::Nothing) = nothing))
      back(Δ) = _gradtuple_kw(_back(Δ))
      return y, back
    end
    nothing
  end
end

macro grad(ex)
  gradm(ex)
end

macro grad!(ex)
  gradm(ex, true)
end

macro nograd(ex)
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = :(;)
  for f in ex.args
    push!(blk.args, :(@inline Zygote._forward(::Context, ::typeof($(esc(f))), args...) = $(esc(f))(args...), Δ -> nothing))
  end
  return blk
end
