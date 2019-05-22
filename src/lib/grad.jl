using MacroTools
using MacroTools: @q, combinedef

named(arg) = isexpr(arg, :(::)) && length(arg.args) == 1 ? :($(gensym())::$(arg.args[1])) : arg

typeless(x) = MacroTools.postwalk(x -> isexpr(x, :(::), :kw) ? x.args[1] : x, x)
isvararg(x) = isexpr(x, :(::)) && namify(x.args[2]) == :Vararg

for n = 0:3
  gradtuple = Symbol(:gradtuple, n)
  @eval begin
    $gradtuple(x::Tuple) = ($(ntuple(_->:nothing,n)...), x...)
    $gradtuple(x::Nothing) = nothing
    $gradtuple(x) = error("Gradient $x should be a tuple")
  end
end

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
  args = named.(args)
  argnames = Any[typeless(arg) for arg in args]
  !isempty(args) && isvararg(args[end]) && (argnames[end] = :($(argnames[end])...,))
  args = esc.(args)
  argnames = esc.(argnames)
  ∂argnames = isclosure ? [f, argnames...] : argnames
  Ts = esc.(Ts)
  cx = :($(esc(:__context__))::Context)
  fargs = kw == nothing ? [cx, :($f::$T), args...] : [kw, cx, :($f::$T), args...]
  gradtuple   = isclosure ? gradtuple0 : gradtuple1
  gradtuplekw = isclosure ? gradtuple2 : gradtuple3
  adj = @q @inline Zygote.adjoint($(fargs...)) where $(Ts...) = $(esc(body))
  quote
    $adj
    @inline function Zygote._forward($cx, $f::$T, $(args...)) where $(Ts...)
      y, _back = adjoint(__context__, $f, $(argnames...))
      $(mut ? nothing : :(back(::Nothing) = nothing))
      back(Δ) = $gradtuple(_back(Δ))
      return y, mutback($cx, $gradtuple(($(∂argnames...),)), y, back)
    end
    @inline function Zygote._forward($cx, ::$kT, kw, $f::$T, $(args...)) where $(Ts...)
      y, _back = adjoint(__context__, $f, $(argnames...); kw...)
      $(mut ? nothing : :(back(::Nothing) = nothing))
      back(Δ) = $gradtuplekw(_back(Δ))
      return y, mutback($cx, $gradtuplekw(($(∂argnames...),)), y, back)
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

macro which(ex)
  @capture(ex, f_(args__)) || error("Zygote.@which f(args...)")
  :(InteractiveUtils.@which adjoint(Context(), $(esc(f)), $(esc.(args)...)))
end
