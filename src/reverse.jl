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

# Only the final BB can return (so we have a single entry point in reverse).
# TODO: merge return nodes
validcfg(ir) =
  ir.stmts[end] isa ReturnNode &&
  !any(x -> x isa ReturnNode, ir.stmts[1:end-1])

# Insert Phi nodes which record branches taken
function record_branches(ir::IRCode)
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

newbi(ir, b) = length(ir.cfg.blocks)-b+1

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

function reverse_ir(ir::IRCode)
  stmts, blocks, grads = [], [], Dict()
  for (i, b) in enumerate(reverse(ir.cfg.blocks))
    preds, succs = newbi.(ir, b.succs), newbi.(ir, b.preds)
    st = length(stmts)+1
    i == 1 && push!(stmts, :(grad()))
    for i = reverse(range(b))
      grad_ex!(stmts, grads, forw[SSAValue(i)], i)
    end
    if isempty(succs)
      push!(stmts, nothing)
    elseif length(succs) == 1
      push!(stmts, GotoNode(succs[1]))
    else
      phi = range(b)[1]
      push!(stmts, GotoIfNot(Alpha(phi), succs[1]))
      push!(stmts, GotoNode(succs[2]))
    end
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
  if all(c -> c in ir.cfg.blocks[b].preds, filter(x -> x != b, bs))
    # TODO: handle the more complex cases here
    @assert length(bs) == length(unique(bs)) == length(ir.cfg.blocks[b].preds)
    PhiNode(bs, gs)
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
    fill_deltas(x::Delta) = fill_delta!(ir, grads, SSAValue(x.id), i)
    fill_deltas(x::Expr) = isexpr(x, :call) ? Expr(:call, fill_deltas.(x.args)...) : x
    ir[SSAValue(i)] = fill_deltas(ir[SSAValue(i)])
  end
  ret = ReturnNode(fill_delta!(ir, grads, Argument(2), length(ir.stmts)))
  insert_node!(ir, length(ir.stmts), Any, ret)
  return ir
end

function grad_ir(ir)
  forw = record_branches(ir)
  rev, grads = reverse_ir(forw)
  return forw, fill_deltas!(rev, grads)
end
