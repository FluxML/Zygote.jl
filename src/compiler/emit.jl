function alphauses(ir, bi)
  us = []
  for i = range(ir.cfg.blocks[bi]), u in userefs(ir.stmts[i])
    u[] isa Alpha && push!(us, SSAValue(u[].id))
  end
  return us
end

xcall(mod::Module, f::Symbol, args...) = Expr(:call, GlobalRef(mod, f), args...)
xcall(f::Symbol, args...) = xcall(Base, f, args...)
xstack(T) = (Vector{T}, Expr(:call, Vector{T}))
xtuple(xs...) = xcall(:tuple, xs...)

afterphi(ir, loc) = ir.stmts[loc] isa PhiNode ? afterphi(ir, loc+1) : loc

function forward_stacks!(adj, F)
  stks, recs = [], []
  for fb = 1:length(adj.perm)
    for α in alphauses(adj.back, adj.perm[fb])
      T = exprtype(adj.forw, α)
      stk = insert_node!(adj.forw, 1, xstack(T)...)
      push!(stks, (adj.perm[fb], alpha(α)))
      push!(recs, stk)
      loc = afterphi(adj.forw, α.id+1)
      insert_node!(adj.forw, loc-1, Any, xcall(:push!, stk, α), true)
    end
  end
  args = [Argument(i) for i = 3:length(adj.forw.argtypes)]
  T = Tuple{exprtype.(Ref(adj.forw), (args..., recs...))...}
  rec = insert_node!(adj.forw, length(adj.forw.stmts), T,
                     xtuple(args..., recs...))
  rec = insert_node!(adj.forw, length(adj.forw.stmts), J{F,T},
                     Expr(:call, J{F}, rec))
  ret = xtuple(adj.forw.stmts[end].val, rec)
  R = exprtype(adj.forw, adj.forw.stmts[end].val)
  ret = insert_node!(adj.forw, length(adj.forw.stmts), Tuple{R,J{F,T}}, ret)
  adj.forw.stmts[end] = ReturnNode(ret)
  forw = compact!(adj.forw)
  return forw, stks
end

function reverse_stacks!(ir, stks, nargs)
  t = insert_node!(ir, 1, Any, xcall(Base, :getfield, Argument(1), QuoteNode(:t)))
  for b = 1:length(ir.cfg.blocks)
    repl = Dict()
    for (i, (b′, α)) in enumerate(stks)
      b == b′ || continue
      loc = max(2,range(ir.cfg.blocks[b])[1])
      stk = insert_node!(ir, loc, Any, xcall(:getindex, t, i+nargs))
      val = insert_node!(ir, loc, Any, xcall(:pop!, stk))
      repl[α] = val
    end
    for i in range(ir.cfg.blocks[b]), u in userefs(ir.stmts[i])
      if u.stmt == Expr(:call, :Δ)
        u.stmt = Argument(2)
      elseif u[] isa Argument
        x = insert_node!(ir, i, Any, xcall(:getindex, t, u[].n-2))
        u[] = x
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
  back = reverse_stacks!(adj.back, stks, length(forw.argtypes)-2)
  return forw, back
end
