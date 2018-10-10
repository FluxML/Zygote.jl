using MacroTools: combinedef

_gradtuple(::Nothing) = nothing
_gradtuple(x::Tuple) = (nothing, x...)
_gradtuple(x) = error("Gradient $x should be a tuple")

function gradm(f, T, args, Ts, body, mut)
  args = esc.(args)
  Ts = esc.(Ts)
  pushfirst!(args, :($(esc(:__context__))::Context), :($f::$T))
  body = quote
    Base.@_inline_meta
    y, back = let
      $(esc(body))
    end
    $(mut ? nothing : :(back2(::Nothing) = nothing))
    # return needed for type inference
    back2(Δ) = return _gradtuple(back(Δ))
    y, back2
  end
  :(Zygote._forward($(args...)) where $(Ts...) = $body; nothing)
end

_gradtuple_kw(::Nothing) = nothing
_gradtuple_kw(x::Tuple) = (nothing, nothing, nothing, x...) # kwfunc, kws, func, args...
_gradtuple_kw(x) = error("Gradient $x should be a tuple")
_untuple_kw(::Nothing) = nothing
_untuple_kw(x::Tuple) = Base.tail(Base.tail(x))

function gradm_kw(f, T, args, Ts, body, mut)
  kws = popfirst!(args)
  Ts = esc.(Ts)
  kT = :(Core.kwftype($T))
  kwargs = [:($(esc(:__context__))::Context), :(::$kT), :kw, f, esc.(args)...]
  kw_wrappers = map(kws.args) do kw
    kw isa Symbol && return :($(esc(kw)) = kw.$kw)
    isexpr(kw, :...) && return :($(esc(kw.args[1])) = kw)
    k, v = kw.args
    :($(esc(k)) = haskey(kw, $(Expr(:quote, k))) ? kw.$k : $(esc(v)))
  end
  body = quote
    Base.@_inline_meta
    $(kw_wrappers...)
    y, back = let
      $(esc(body))
    end
    $(mut ? nothing : :(back2(::Nothing) = nothing))
    # return needed for type inference
    back2(Δ) = return _gradtuple_kw(back(Δ))
    y, back2
  end
  quote
    Zygote._forward($(kwargs...)) where $(Ts...) = $body
    function Zygote._forward(cx::Context, f::$T, $(esc.(args)...)) where $(Ts...)
      y, back = _forward(cx, Core.kwfunc(f), NamedTuple(), f, $(esc.(namify.(args))...)) # TODO unnamed arguments
      return y, Δ -> _untuple_kw(back(Δ))
    end
    nothing
  end
end

function gradm(ex, mut = false)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {Ts__} = body_)) || error("Need a function definition")
  iskw = length(args) > 1 && isexpr(args[1], :parameters)
  name, T = isexpr(name, :(::)) ?
    (length(name.args) == 1 ? (:_, esc(name.args[1])) : esc.(name.args)) :
    (:_, :(typeof($(esc(name)))))
  Ts == nothing && (Ts = [])
  return (iskw ? gradm_kw : gradm)(name, T, args, Ts, body, mut)
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
