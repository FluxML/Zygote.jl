using IRTools: varargs!, inlineable!, pis!, slots!
using IRTools.Inner: argnames!, update!

ignore_sig(T) = all(T -> T <: Type, T.parameters)


function _pullback(ctx::AContext, f, args...)
  if chainrules_blacklist(f, args...)
    # then don't even consider using ChainRules
    return _pullback_via_source2source(ctx, f, args...)
  end

  res = ChainRules.rrule(f, args...)
  if res === nothing
    # No ChainRule defined, time to do the source tranform
    return _pullback_via_source2source(ctx, f, args...)
  else
    # Can just use ChainRule answer
    y, pb = res
    return y, _pullback_via_chainrules(pb)
  end
end

#=="""
  chainrules_blacklist(f, args...,)

This is used to disable the use of ChainRule's definitions
for particular functions/methods.

It is not required if a Zygote rule has already been defined directly.
"""==#
chainrules_blacklist(f, args...) = false

# ChainRules does higher-order functions badly
# see https://github.com/JuliaDiff/ChainRules.jl/issues/122
for f in (map, broadcast, mapreduce, mapfoldl, mapfoldr)
    @eval chainrules_blacklist(::typeof($f), args...) = true
end
chainrules_blacklist(::typeof(sum), f, x::AbstractArray{<:Real}) = true
# Except for sum(abs2, xs), that is fine
chainrules_blacklist(::typeof(sum), ::typeof(abs2), x::AbstractArray{<:Real}) = false

# ChainRules current Wirtinger deriviative is not compatible
# reconsider after https://github.com/JuliaDiff/ChainRulesCore.jl/pull/29
for f in (abs, abs2, conj, adjoint, hypot, angle, imag, real)
  @eval chainrules_blacklist(::typeof($f), ::Complex) = true
end

# Sum of nonarrays doesn't really work
# Fixed in https://github.com/JuliaDiff/ChainRules.jl/pull/124
chainrules_blacklist(::typeof(sum), x) = true
chainrules_blacklist(::typeof(sum), x::AbstractArray{<:Real}) = false


#=="""
  _pullback_via_chainrules(pb)

Converts a ChainRules pullback into a Zygote pullback.
`pb` should be a ChainRules pullback, as returned from the second return value of `rrule`
"""==#
function _pullback_via_chainrules(pb)
  # This is the optimized version of
  # _pullback_via_chainrules(pb) = (Δs...) -> zextern.(pb(Δs...))
  function zback(Δs...)
    ∂s = pb(Δs...)
    ntuple(length(∂s)) do ii
      ∂ = ∂s[ii]
      zextern(∂)
    end
  end
end

zextern(x) = ChainRules.extern(x)
zextern(::ChainRules.Zero) = nothing  # Zygote loves calling things nothing
zextern(::ChainRules.DNE) = nothing  # Zygote loves calling things nothing


@generated function _pullback_via_source2source(ctx::AContext, f, args...)
  T = Tuple{f,args...}
  ignore_sig(T) && return :(f(args...), Pullback{$T}(()))
  g = try _lookup_grad(T) catch e e end
  !(g isa Tuple) && return :(f(args...), Pullback{$T}((f,)))
  meta, forw, _ = g
  argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(meta, forw, 3)
  # IRTools.verify(forw)
  forw = slots!(pis!(inlineable!(forw)))
  return update!(meta.code, forw)
end

@generated function (j::Pullback{T})(Δ) where T
  ignore_sig(T) && return :nothing
  g = try _lookup_grad(T)
  catch e
    rethrow(CompileError(T,e))
  end
  if g == nothing
    Δ == Nothing && return :nothing
    return :(error("Non-differentiable function $(repr(j.t[1]))"))
  end
  meta, _, back = g
  argnames!(meta, Symbol("#self#"), :Δ)
  # IRTools.verify(back)
  back = slots!(inlineable!(back))
  return update!(meta.code, back)
end
