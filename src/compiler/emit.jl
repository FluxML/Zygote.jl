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

function do_dfs(cfg, bb)
    # TODO: There are algorithms to this with much better
    # asymptotics, but this'll do for now.
    visited = Set{Int}(bb)
    worklist = Int[bb]
    while !isempty(worklist)
      block = pop!(worklist)
      for succ in cfg.blocks[block].succs
        succ == bb && return nothing
        (succ in visited) && continue
        push!(visited, succ)
        push!(worklist, succ)
      end
    end
    return collect(visited)
end

function find_dominating_for_bb(domtree, bb, alpha_ssa, def_bb, phi_nodes)
  # Ascend idoms, until we find another phi block or the def block
  while bb != 0
    if bb == def_bb
      return alpha_ssa
    elseif haskey(phi_nodes, bb)
      return phi_nodes[bb][2]
    end
    bb = domtree.idoms[bb]
  end
  return nothing
end

function insert_phi_nest!(ir, domtree, T, def_bb, exit_bb, alpha_ssa, phi_blocks)
  phi_nodes = Dict(bb => (pn = PhiNode(); ssa = insert_node!(ir, first(ir.cfg.blocks[bb].stmts), Union{T, Nothing}, pn); (pn, ssa)) for bb in phi_blocks)
  # TODO: This could be more efficient by joint ascension of the domtree
  for bb in phi_blocks
    bb_phi, _ = phi_nodes[bb]
    for pred in ir.cfg.blocks[bb].preds
      dom = find_dominating_for_bb(domtree, pred, alpha_ssa, def_bb, phi_nodes)
      if dom !== nothing
        push!(bb_phi.edges, pred)
        push!(bb_phi.values, dom)
      else
        push!(bb_phi.edges, pred)
        push!(bb_phi.values, nothing)
      end
    end
  end
  exit_dom = find_dominating_for_bb(domtree, exit_bb, alpha_ssa, def_bb, phi_nodes)
end

function forward_stacks!(adj, F)
  stks, recs = [], []
  fwd_cfg = adj.forw.cfg
  domtree = construct_domtree(fwd_cfg)
  exit_bb = length(fwd_cfg.blocks)
  for fb in adj.perm
    # TODO: do_dfs does double duty here, computing self reachability and giving
    # us the set of live in nodes. There are better algorithms for the former
    # and the latter shouldn't be necessary.
    live_in = do_dfs(fwd_cfg, fb)
    in_loop = live_in === nothing
    if !in_loop
      if dominates(domtree, fb, exit_bb)
        phi_blocks = Int[]
      else
        # Liveness is trivial here, so we could specialize idf
        # on that fact, but good enough for now.
        phi_blocks = Core.Compiler.idf(fwd_cfg, Core.Compiler.BlockLiveness([fb], live_in), domtree)
      end
    end
    for α in alphauses(adj.back, invperm(adj.perm)[fb])
      T = exprtype(adj.forw, α)
      if !in_loop
        α′ = insert_phi_nest!(adj.forw, domtree, T, fb, exit_bb, SSAValue(α.id), phi_blocks)
        pushfirst!(recs, α′)
      else
        stk = insert_node!(adj.forw, 1, xstack(T)...)
        pushfirst!(recs, stk)
        insert_blockend!(adj.forw, blockidx(adj.forw, α.id), Any, xcall(Zygote, :_push!, stk, α))
      end
      pushfirst!(stks, (invperm(adj.perm)[fb], alpha(α), in_loop))
    end
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

# If we had the type, we could make this a PiNode
notnothing(x::Nothing) = error()
notnothing(x) = x

function reverse_stacks!(adj, stks)
  ir = adj.back
  t = insert_node!(ir, 1, Any, xcall(Base, :getfield, Argument(1), QuoteNode(:t)))
  for b = 1:length(ir.cfg.blocks)
    repl = Dict()
    for (i, (b′, α, use_stack)) in enumerate(stks)
      b == b′ || continue
      loc, attach_after = afterphi(ir, range(ir.cfg.blocks[b])[1])
      loc = max(2, loc)
      if !use_stack
        val = insert_node!(ir, loc, Any, xcall(:getindex, t, i), attach_after)
        val = insert_node!(ir, loc, Any, xcall(Zygote, :notnothing, val), attach_after)
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
