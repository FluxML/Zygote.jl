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

xstack(T) = stmt(Expr(:call, Vector{T}), type = Vector{T})

function alphauses(b)
  us = Set{Alpha}()
  postwalk(x -> x isa Alpha && push!(us, x), b)
  return us
end

xtuple(xs...) = xcall(:tuple, xs...)

concrete(T::DataType) = T
concrete(::Type{Type{T}}) where T = typeof(T)
concrete(T) = Any

# function stacklines(adj::Adjoint)
#   recs = []
#   for fb in adj.perm, α in alphauses(adj.back, invperm(adj.perm)[fb])
#     pushfirst!(recs, adj.forw.linetable[adj.forw.lines[α.id]])
#   end
#   return recs
# end

runonce(b) = b.id in (1, length(b.ir.blocks))

function forward_stacks!(adj, F)
  stks, recs = [], []
  pr = adj.primal
  for b in blocks(pr), α in alphauses(block(adj.adjoint, b.id))
    if runonce(b)
      push!(recs, Variable(α))
    else
      T = exprtype(pr, Variable(α))
      stk = pushfirst!(pr, xstack(T))
      push!(recs, stk)
      push!(b, xcall(Zygote, :_push!, stk, Variable(α)))
    end
    push!(stks, (b.id, alpha(α)))
  end
  args = [arg(i) for i = 3:length(pr.args)]
  T = Tuple{concrete.(exprtype.((pr,), recs))...}
  isconcretetype(T) || (T = Any)
  rec = push!(pr, xtuple(recs...))
  if usetyped && length(pr.blocks) > 1
    rec = push!(pr, Expr(:call, Pullback{F,T}, rec))
  else
    P = length(pr.blocks) == 1 ? Pullback{F} : Pullback{F,Any}
    rec = push!(pr, Expr(:call, P, rec))
  end
  ret = xtuple(pr.blocks[end].branches[end].args[1], rec)
  ret = push!(pr, ret)
  pr.blocks[end].branches[end].args[1] = ret
  return pr, stks
end

function reverse_stacks!(adj, stks)
  ir = adj.adjoint
  t = pushfirst!(blocks(ir)[end], xcall(:getfield, Argument(1), QuoteNode(:t)))
  entry = blocks(ir)[end]
  repl = Dict()
  runonce(b) = b.id in (1, length(ir.blocks))
  for b in blocks(ir)
    for (i, (b′, α)) in enumerate(stks)
      b.id == b′ || continue
      if runonce(b)
        val = insertafter!(ir, t, xcall(:getindex, t, i))
      else
        stk = push!(entry, xcall(:getindex, t, i))
        stk = push!(entry, xcall(Zygote, :Stack, stk))
        val = pushfirst!(b, xcall(:pop!, stk))
      end
      repl[α] = val
    end
  end
  repl[arguments(entry)[1]] = arg(2)
  empty!(arguments(entry))
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

meta(T) = (usetyped ? IRTools.typed_meta : IRTools.meta)(T)

function getmeta(T)
  m = meta(T)
  (usetyped && m != nothing) || return m
  any(x -> isexpr(x, :goto, :gotoifnot), m.code.code) || return IRTools.meta(T)
  return m
end

function _lookup_grad(T)
  (m = getmeta(T)) == nothing && return
  m isa IRTools.TypedMeta && m.ret == Union{} && return
  va = varargs(m.method, length(T.parameters))
  forw, back = stacks!(Adjoint(IR(m), varargs = va, normalise = false), T)
  m, forw, back
end

stacklines(T::Type) = stacklines(Adjoint(IR(meta(T))))
