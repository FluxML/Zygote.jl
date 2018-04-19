struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::SSAValue) = Alpha(x.id)

struct Delta
  id::Int
end

Base.show(io::IO, x::Delta) = print(io, "Δ", x.id)

# TODO: merge return nodes
validcfg(ir) =
  ir.stmts[end] isa ReturnNode &&
  !any(x -> x isa ReturnNode, ir.stmts[1:end-1])

function record_branches!(ir::IRCode)
  ir = IncrementalCompact(ir)
  for (i, x) in ir
    bi = findfirst(x -> x == i+1, ir.ir.cfg.index)
    bi == nothing && continue
    preds = ir.ir.cfg.blocks[bi+1].preds
    length(preds) > 1 || continue
    @assert length(preds) <= 2
    insert_node_here!(ir, PhiNode(preds, [false, true]), Bool, ir.result_lines[i])
  end
  return finish(ir)
end

function reachable(ir)
  seen = SSAValue[]
  stack = [SSAValue(length(ir.stmts))]
  while !isempty(stack)
    i = popfirst!(stack)
    i ∈ seen && continue
    push!(seen, i)
    NI.foreachssa(x -> push!(stack, x), ir[i])
  end
  return seen
end

function grad_ex!(stmts, grads, ex, i)
  grad(x, Δ) = push!(get!(grads, x, []), Δ)
  grad(x::Union{Argument,SSAValue}) = grad(x, SSAValue(length(stmts)))
  grad(x) = return
  if ex isa Union{GotoNode,GotoIfNot,Void}
  elseif ex isa ReturnNode
    grad(ex.val, SSAValue(1))
  elseif ex isa PhiNode
    push!(stmts, Delta(i))
    grad.(ex.values)
  elseif isexpr(ex, :call)
    args = ex.args[2:end]
    push!(stmts, Expr(:call, :∇, ex.args[1], Delta(i), alpha.(args)...))
    Δ = SSAValue(length(stmts))
    for (i, x) in enumerate(args)
      push!(stmts, Expr(:call, GlobalRef(Base, :getindex), Δ, i))
      grad(x)
    end
  else
    error("Can't handle $ex")
  end
end

function reverse_cfg(cfg, perm)
  newidx(i) = perm[i]
  CFG([BasicBlock(StmtRange(1,0),newidx.(b.succs),newidx.(b.preds)) for b in cfg.blocks[perm]])
end

function reverse_order(cfg)
  n = length(cfg.blocks)
  perm = n:-1:1
  guess = reverse_cfg(cfg, perm)
  dt = construct_domtree(guess)
  perm[sortperm(1:n, by = x -> dt.nodes[x].level)]
end

function reverse_ir(ir::IRCode)
  stmts, blocks, grads = [], [], Dict()
  valid = reachable(ir)
  perm = reverse_order(ir.cfg)
  newidx(i) = perm[i]
  for (bi, b) in enumerate(ir.cfg.blocks[perm])
    preds, succs = newidx.(b.succs), newidx.(b.preds)
    st = length(stmts)+1
    bi == 1 && push!(stmts, Expr(:call, :Δ))
    for i = reverse(range(b))
      SSAValue(i) in valid || continue
      grad_ex!(stmts, grads, ir[SSAValue(i)], i)
    end
    if isempty(succs)
    elseif length(succs) == 1
      push!(stmts, GotoNode(succs[1]))
    else
      phi = range(b)[1]
      push!(stmts, GotoIfNot(Alpha(phi), succs[1]))
      push!(stmts, GotoNode(succs[2]))
    end
    bi == length(ir.cfg.blocks) && push!(stmts, ReturnNode(nothing))
    push!(blocks, BasicBlock(StmtRange(st,length(stmts)), preds, succs))
  end
  rev = IRCode(ir, stmts, Any[Any for _ in stmts], [-1 for _ in stmts], CFG(blocks), NI.NewNode[])
  return rev, grads
end

accum_symbolic(gs) = reduce((a,b) -> :($(GlobalRef(Base,:+))($a,$b)), gs)

function fill_delta!(ir, grads, x, i)
  haskey(grads, x) || return nothing
  dt = construct_domtree(ir.cfg)
  gs = grads[x]
  b, bs = blockidx(ir, i), blockidx.(ir, gs)
  if all(c -> c == b, bs)
    accum_symbolic(gs)
  elseif all(c -> c in ir.cfg.blocks[b].preds, bs)
    # TODO: handle the more complex cases here
    @assert length(bs) == length(unique(bs)) == length(ir.cfg.blocks[b].preds)
    length(bs) == 1 ? gs[1] : PhiNode(bs, gs)
  else
    # TODO: find a common dominator
    Δ = insert_node!(ir, 2, Any, Expr(:call, GlobalRef(Base, :similar), alpha(x)))
    for g in gs
      insert_node!(ir, g.id+1, Any, Expr(:call, :accum!, Δ, g))
    end
    Δ
  end
end

function fill_deltas!(ir, grads)
  for i = 1:length(ir.stmts)
    # TODO: use userefiterator stuff for this
    fill_deltas(x) = x
    fill_deltas(x::Expr) = isexpr(x, :call) ? Expr(:call, fill_deltas.(x.args)...) : x
    fill_deltas(x::Delta) = fill_delta!(ir, grads, SSAValue(x.id), i)
    fill_deltas(x::ReturnNode) = ReturnNode(fill_delta!(ir, grads, Argument(2), i))
    ir[SSAValue(i)] = fill_deltas(ir[SSAValue(i)])
  end
  return compact!(ir)
end

function grad_ir(ir)
  validcfg(ir) || error("Multiple return not supported")
  forw = record_branches!(ir)
  rev, grads = reverse_ir(forw)
  return forw, fill_deltas!(rev, grads)
end
