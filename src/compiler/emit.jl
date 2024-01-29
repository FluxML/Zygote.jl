# Stacks

mutable struct Stack{T}
  idx::Int
  data::Vector{T}
end

Stack(data::Vector{T}) where T =
  Stack{T}(length(data), data)

function Base.pop!(stk::Stack)
  i = stk.idx
  stk.idx = i == 1 ? length(stk.data) : i-1
  @inbounds return stk.data[i]
end

function _push!(a::Vector{T}, x::T) where T
  Base._growend!(a, 1)
  @inbounds a[end] = x
  return
end

# Emit

xstack(T) = Expr(:call, Vector{T})

function alphauses(b)
  us = Set{Alpha}()
  postwalk(x -> x isa Alpha && push!(us, x), b)
  return us
end

xtuple(xs...) = xcall(:tuple, xs...)

concrete(T::DataType) = T
concrete(::Type{Type{T}}) where T = typeof(T)
concrete(T) = Any

runonce(b) = b.id in (1, length(b.ir.blocks)) &&
             !any(((_,stmt),) -> isexpr(stmt.expr, :catch), b)

function forward_stacks!(adj, F)
  stks, recs = [], []
  pr = adj.primal
  for b in blocks(pr), α in alphauses(block(adj.adjoint, b.id))
    not_stack = runonce(b)
    if not_stack
      push!(recs, Variable(α))
    else
      stk = pushfirst!(pr, xstack(Any))
      push!(recs, stk)
      push!(b, xcall(Zygote, :_push!, stk, Variable(α)))
    end
    push!(stks, (b.id, alpha(α), not_stack))
  end
  rec = push!(pr, xtuple(recs...))
  P = length(pr.blocks) == 1 ? Pullback{F} : Pullback{F,Any}
  # P = Pullback{F,Any} # reduce specialisation
  rec = push!(pr, Expr(:call, P, rec))
  ret = xtuple(pr.blocks[end].branches[end].args[1], rec)
  ret = push!(pr, ret)
  pr.blocks[end].branches[end].args[1] = ret
  return pr, stks
end

function reverse_stacks!(adj, stks)
  ir = adj.adjoint
  entry = blocks(ir)[end]
  self = argument!(entry, at = 1)
  t = pushfirst!(blocks(ir)[end], xcall(:getfield, self, QuoteNode(:t)))
  repl = Dict()
  for b in blocks(ir)
    for (i, (b′, α, not_stack)) in enumerate(stks)
      b.id == b′ || continue
      if not_stack
        val = insertafter!(ir, t, xcall(:getindex, t, i))
      else
        stk = push!(entry, xcall(:getindex, t, i))
        stk = push!(entry, xcall(Zygote, :Stack, stk))
        val = pushfirst!(b, xcall(:pop!, stk))
      end
      repl[α] = val
    end
  end
  return IRTools.prewalk!(x -> get(repl, x, x), ir)
end

function stacks!(adj, T)
  forw, stks = forward_stacks!(adj, T)
  back = reverse_stacks!(adj, stks)
  permute!(back, length(back.blocks):-1:1)
  IRTools.domorder!(back)
  return forw, back
end

varargs(m::Method, n) = m.isva ? n - m.nargs + 1 : nothing

function _generate_pullback_via_decomposition(T, world)
  (m = meta(T; world)) === nothing && return
  va = varargs(m.method, length(T.parameters))
  forw, back = stacks!(Adjoint(IR(m), varargs = va, normalise = false), T)
  m, forw, back
end

function stacklines(T::Type)
  adj = Adjoint(IR(meta(T)), normalise = false)
  recs = []
  for b in blocks(adj.adjoint), α in alphauses(b)
    push!(recs, IRTools.exprline(adj.primal, Variable(α)))
  end
  return recs
end
