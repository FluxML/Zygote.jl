using IRTools.All
using IRTools: Pipe
using Base: tail

ntail(x, n) = n <= 0 ? x : xcall(:tail, ntail(x, n-1))

function instrument!(pr, v, st)
  ex = st.expr
  if isexpr(ex, :new)
    st = stmt(st, expr = xcall(Zygote, :__new__, ex.args...))
    pr[v] = st
  elseif isexpr(ex, :splatnew)
    st = stmt(st, expr = xcall(Zygote, :__splatnew__, ex.args...))
    pr[v] = st
  end
  return st
end

function dual(ir)
  args = copy(arguments(ir))
  dx = argument!(ir, at = 1)
  Δs = Dict()
  for bl in blocks(ir)[2:end], arg in copy(arguments(bl))
    Δs[arg] = argument!(bl, insert = false)
  end
  pr = Pipe(ir)
  partial(x::Variable) = Δs[x]
  partial(x) = push!(pr, xcall(Forward, :zerolike, x))
  partial(v, x::Variable) = Δs[x]
  partial(v, x) = insert!(pr, v, xcall(Forward, :zerolike, x))
  for (i, x) in enumerate(args)
    if i == length(args) && ir.meta.method.isva
      Δs[x] = push!(pr, ntail(dx, i-1))
    else
      Δs[x] = push!(pr, xcall(:getindex, dx, i))
    end
  end
  branches(pr) do br
    args = arguments(br)
    if isreturn(br)
      args[1] = push!(pr, xcall(:tuple, args[1], partial(args[1])))
    else
      for arg in copy(args)
        push!(args, partial(arg))
      end
    end
    br
  end
  for (v, st) in pr
    st = instrument!(pr, v, st)
    if isexpr(st.expr, :meta, :inbounds, :loopinfo)
      Δs[v] = nothing
    elseif isexpr(st.expr, :boundscheck) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Base, :not_int)) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Core, :(===))) ||
           (isexpr(st.expr, :call) && st.expr.args[1] == GlobalRef(Main, :(===)))
      Δs[v] = false
    elseif isexpr(st.expr, :call)
      dargs = insert!(pr, v, xcall(:tuple, partial.((v,), st.expr.args)...))
      result = insert!(pr, v, stmt(st, expr = xcall(Forward, :_pushforward, dargs, st.expr.args...)))
      pr[v] = xcall(:getindex, result, 1)
      Δs[v] = push!(pr, xcall(:getindex, result, 2))
    elseif !isexpr(st.expr)
      Δs[v] = push!(pr, xcall(Forward, :zerolike, v))
    else
      error("Unsupported $(st.expr.head) expression")
    end
  end
  ir = finish(pr)
  return ir
end

const chainrules_frule_fallback = which(frule, Tuple{Any})

"""
  has_chain_frule(T)

For a type-tuple `T` e.g. `Tuple{typeof(f), Int, Float64}`, checks if there is a `rrule` defined for it.
Excluding the generic fallback.
The first return value is `true` if the `rrule` exists, `false` otherwise.
If it does not, then the second argument is a list of edges to attach to the CodeInfo for a generated function,
such that if a suitable rule is defined later, the generated function will recompile.
"""
function has_chain_frule(T)
  m = meta(Tuple{typeof(frule), T.parameters...})

  if m.method !== chainrules_frule_fallback
    # found a frule, no need to add any edges
    return true, nothing
  end

  return false, m.instance
end

"""
    chain_frule(f, args...)

Returns a the (primal) value of `f(args...)` and tangent, by invoking
`ChainRules.frule(f, args...)`.
"""
@noinline chain_frule(dargs, args...) = frule(dargs, args...)

"""
  chain_frule_kw(kwf, kwargs, f, args...)

As per [`chain_frule`](@ref) but with support for kwargs.
`kwf` should be the kwfunc matching to `f`, and `kwargs` are a `NamedTuple` of keyword
arguments.
"""
@inline chain_frule_kw(kwf, kwargs, f, args...) = frule(f, args...; kwargs...)

ignore_sig(T) = all(T -> T <: Type, T.parameters)

"""
    is_kwfunc(sigt...)

Determines if `sigt` is the type signature of a kwfunction.
Each element of `sigt` should be a type.
Either the first 3 types are a kwfunc type, a NamedTuple and the matching base function type,
or the first argument is the base function type and it is not a kwfunction.
the remaining types in `sigt` are the types of the argument.

"""
is_kwfunc(::Vararg{Any}) = false
is_kwfunc(k, ::Type{<:NamedTuple}, f, args...) = k===Core.kwftype(f)

@generated function _pushforward(dargs, f, args...)
  T = Tuple{dargs, f, args...}
  ignore_sig(T) && return :(f(args...), Zero())

  iskw = is_kwfunc(dargs, f, args...)
  # if it is_kw then `args[1]` are the keyword args, `args[2]` is actual function
  base_T = iskw ? Tuple{args[2:end]...} : T
  hascr, cr_edge = has_chain_frule(base_T)

  chain_frule_f = iskw ? :chain_frule_kw : :chain_frule
  if hascr
    return :($chain_frule_f(dargs, f, args...))
  else
    return :(__pushforward(dargs, f, args...))
  end

  # g = try _lookup_grad(T) catch e e end
  # !(g isa Tuple) && return :(f(args...), Pullback{$T}((f,)))
  # meta, forw, _ = g
  # argnames!(meta, Symbol("#self#"), :ctx, :f, :args)
  # forw = varargs!(meta, forw, 3)
  # # IRTools.verify(forw)
  # forw = slots!(pis!(inlineable!(forw)))
  # @static if VERSION >= v"1.3" # no edges pre-1.3
  #   # be ready to swap to using chainrule if one is declared
  #   cr_edge != nothing && edge!(meta, cr_edge)
  # end
  # return update!(meta.code, forw)
end

@dynamo function __pushforward(_, x...)
  ir = IR(x...)
  ir == nothing && return :(error("non-differentiable function $(args[2])"))
  ir = Zygote.instrument(ir)
  ir.meta.code.inlineable = true
  return dual(ir)
end
