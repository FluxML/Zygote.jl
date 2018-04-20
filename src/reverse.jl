struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::SSAValue) = Alpha(x.id)

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
  seen = []
  stack = [ir.stmts[end].val]
  while !isempty(stack)
    i = popfirst!(stack)
    i ∈ seen && continue
    push!(seen, i)
    for x in userefs(ir[i])
      x[] isa SSAValue && push!(stack, x[])
      x[] isa Argument && x[] ∉ seen && push!(seen, x[])
    end
  end
  return seen
end

accumm!(x, Δ) = Expr(:call, GlobalRef(Zygote, :accum!), x, Δ)

function grad_ex!(stmts, grads, ex, i)
  if ex isa Union{GotoNode,GotoIfNot,Void}
  elseif ex isa ReturnNode
    push!(stmts, accumm!(grads[ex.val], SSAValue(1)))
  elseif isexpr(ex, :call)
    args = ex.args[2:end]
    push!(stmts, Expr(:call, GlobalRef(Zygote, :∇), ex.args[1], grads[SSAValue(i)], alpha.(args)...))
    Δ = SSAValue(length(stmts))
    for (i, x) in enumerate(args)
      haskey(grads, x) || continue
      push!(stmts, Expr(:call, GlobalRef(Base, :getindex), Δ, i))
      push!(stmts, accumm!(grads[x], SSAValue(length(stmts))))
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
  stmts, blocks, phis, grads = [], [], [], Dict()
  perm = reverse_order(ir.cfg)
  newidx(i) = perm[i]
  push!(stmts, :(Δ()))
  # TODO: put these in the right block
  for x in reachable(ir)
    push!(stmts, Expr(:call, GlobalRef(Zygote, :grad), alpha(x)))
    grads[x] = SSAValue(length(stmts))
  end
  for (bi, b) in enumerate(ir.cfg.blocks[perm])
    preds, succs = newidx.(b.succs), newidx.(b.preds)
    st = bi == 1 ? 1 : length(stmts)+1
    for (from, to, x, Δ) in phis
      bi == to || continue
      @assert length(preds) == 1
      push!(stmts, accumm!(grads[x], grads[Δ]))
    end
    for i = reverse(range(b))
      i == length(ir.stmts) || haskey(grads, SSAValue(i)) || continue
      ex = ir[SSAValue(i)]
      if ex isa PhiNode
        for (e, v) in zip(ex.edges, ex.values)
          haskey(grads, v) || continue
          push!(phis, (bi, newidx(e), v, SSAValue(i)))
        end
      else
        grad_ex!(stmts, grads, ir[SSAValue(i)], i)
      end
    end
    if isempty(succs)
    elseif length(succs) == 1
      push!(stmts, GotoNode(succs[1]))
    else
      phi = range(b)[1]
      push!(stmts, GotoIfNot(Alpha(phi), succs[1]))
      push!(stmts, GotoNode(succs[2]))
    end
    bi == length(ir.cfg.blocks) && push!(stmts, ReturnNode(grads[Argument(2)]))
    push!(blocks, BasicBlock(StmtRange(st,length(stmts)), preds, succs))
  end
  rev = IRCode(ir, stmts, Any[Any for _ in stmts], [-1 for _ in stmts], CFG(blocks), NI.NewNode[])
  return rev
end

function grad_ir(ir)
  validcfg(ir) || error("Multiple return not supported")
  forw = record_branches!(ir)
  back = reverse_ir(forw)
  return forw, back
end
