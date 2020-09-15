using MacroTools: @capture, @q, shortdef
using ZygoteRules: named, typeless, isvararg
using Base: tail

drop(x, n) = n == 0 ? x : :(tail($(drop(x, n-1))))
drop(n) = x -> drop(x, n)

# TODO: move to ZygoteRules
function tangent end

function gradm(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {Ts__} = body_)) || error("Need a function definition")
  kw = length(args) > 1 && isexpr(args[1], :parameters) ? esc(popfirst!(args)) : nothing
  isclosure = isexpr(name, :(::)) && length(name.args) > 1
  f, T = isexpr(name, :(::)) ?
    (length(name.args) == 1 ? (esc(gensym()), esc(name.args[1])) : esc.(name.args)) :
    (esc(gensym()), :(Core.Typeof($(esc(name)))))
  kT = :(Core.kwftype($T))
  Ts === nothing && (Ts = [])
  args = named.(args)
  argnames = Any[typeless(arg) for arg in args]
  !isempty(args) && isvararg(args[end]) && (argnames[end] = :($(argnames[end])...,))
  args = esc.(args)
  argnames = esc.(argnames)
  Ts = esc.(Ts)
  fargs = kw === nothing ? [:($f::$T), args...] : [kw, :($f::$T), args...]
  dropg  = isclosure ? identity : drop(1)
  dropkw = isclosure ?  drop(2) : drop(3)
  adj = @q @inline Zygote.Forward.tangent($(fargs...)) where $(Ts...) = $(esc(body))
  quote
    $adj
    @inline function Zygote.Forward._pushforward(partials, $f::$T, $(args...)) where $(Ts...)
      y, forw = tangent($f, $(argnames...))
      return y, forw($(dropg(:partials))...)
    end
    @inline function Zygote.Forward._pushforward(dargs, ::$kT, kw, $f::$T, $(args...)) where $(Ts...)
      y, forw = tangent($f, $(argnames...))
      return y, forw($(dropkw(:partials))...)
    end
    nothing
  end
end

macro tangent(ex)
  gradm(ex)
end

pushforward(f, x...) = (ẋ...) -> _pushforward((zerolike(f), ẋ...), f, x...)[2]
