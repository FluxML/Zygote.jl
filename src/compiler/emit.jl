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

xstack(T) = (Vector{T}, Expr(:call, Vector{T}))

function _push!(a::Vector{T}, x::T) where T
  Base._growend!(a, 1)
  @inbounds a[end] = x
  return
end

# Emit

function alphauses(ir, bi)
  us = []
  for i = range(ir.cfg.blocks[bi]), u in userefs(ir.stmts[i])
    u[] isa Alpha && push!(us, SSAValue(u[].id))
  end
  return unique(us)
end

xtuple(xs...) = xcall(:tuple, xs...)

concrete(T::DataType) = T
concrete(::Type{Type{T}}) where T = typeof(T)
concrete(T) = Any

function stacklines(adj::Adjoint)
  recs = []
  for fb in adj.perm, α in alphauses(adj.back, invperm(adj.perm)[fb])
    pushfirst!(recs, adj.forw.linetable[adj.forw.lines[α.id]])
  end
  return recs
end

function forward_stacks!(adj, F)
  stks, recs = [], []
  for fb in adj.perm, α in alphauses(adj.back, invperm(adj.perm)[fb])
    if fb == 1
      pushfirst!(recs, α)
    else
      T = exprtype(adj.forw, α)
      stk = insert_node!(adj.forw, 1, xstack(T)...)
      pushfirst!(recs, stk)
      insert_blockend!(adj.forw, blockidx(adj.forw, α.id), Any, xcall(Zygote, :_push!, stk, α))
    end
    pushfirst!(stks, (invperm(adj.perm)[fb], alpha(α)))
  end
  args = [Argument(i) for i = 3:length(adj.forw.argtypes)]
  T = Tuple{concrete.(exprtype.((adj.forw,), recs))...}
  isconcretetype(T) || (T = Any)
  rec = insert_node!(adj.forw, length(adj.forw.stmts), T,
                     xtuple(recs...))
  if usetyped
    rec = insert_node!(adj.forw, length(adj.forw.stmts), Pullback{F,T},
                       Expr(:call, Pullback{F,T}, rec))
  else
    rec = insert_node!(adj.forw, length(adj.forw.stmts), Any,
                       Expr(:call, Pullback{F}, rec))
  end
  ret = xtuple(adj.forw.stmts[end].val, rec)
  R = exprtype(adj.forw, adj.forw.stmts[end].val)
  ret = insert_node!(adj.forw, length(adj.forw.stmts), Tuple{R,Pullback{F,T}}, ret)
  adj.forw.stmts[end] = ReturnNode(ret)
  forw = compact!(adj.forw)
  return forw, stks
end

function reverse_stacks!(adj, stks)
  ir = adj.back
  t = insert_node!(ir, 1, Any, xcall(Base, :getfield, Argument(1), QuoteNode(:t)))
  for b = 1:length(ir.cfg.blocks)
    repl = Dict()
    for (i, (b′, α)) in enumerate(stks)
      b == b′ || continue
      loc, attach_after = afterphi(ir, range(ir.cfg.blocks[b])[1])
      loc = max(2, loc)
      if adj.perm[b′] == 1
        val = insert_node!(ir, loc, Any, xcall(:getindex, t, i), attach_after)
      else
        stk = insert_node!(ir, 1, Any, xcall(:getindex, t, i))
        stk = insert_node!(ir, 1, Any, xcall(Zygote, :Stack, stk))
        val = insert_node!(ir, loc, Any, xcall(:pop!, stk), attach_after)
      end
      repl[α] = val
    end
    for i in range(ir.cfg.blocks[b]), u in userefs(ir.stmts[i])
      if u.stmt == Expr(:call, :Δ)
        u.stmt = Argument(2)
      elseif haskey(repl, u[])
        u[] = repl[u[]]
      else continue
      end
      ir.stmts[i] = u.stmt
    end
  end
  return compact!(ir)
end

function stacks!(adj, T)
  forw, stks = forward_stacks!(adj, T)
  back = reverse_stacks!(adj, stks)
  return forw, back
end

varargs(m::Method, n) = m.isva ? n - m.nargs + 1 : nothing

function _lookup_grad(T)
  (m = meta(T)) == nothing && return
  usetyped && m.ret == Union{} && return
  va = varargs(m.method, length(T.parameters))
  forw, back = stacks!(Adjoint(IRCode(m), varargs = va), T)
  # verify_ir(forw)
  # verify_ir(back)
  m, forw, back
end

stacklines(T::Type) = stacklines(Adjoint(IRCode(meta(T))))
