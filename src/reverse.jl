struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::SSAValue) = Alpha(x.id)

xgetindex(x, i...) = Expr(:call, GlobalRef(Base, :getindex), x, i...)

xaccum!(x, Δ) = Expr(:call, GlobalRef(Zygote, :accum!), x, Δ)

const x∇ = GlobalRef(Zygote, :∇)

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

function record!(ir::IRCode)
  xs = reachable(ir)
  for i = 1:length(ir.stmts)
    ex = ir[SSAValue(i)]
    (SSAValue(i) ∈ xs && isexpr(ex, :call)) || continue
    yJ = insert_node!(ir, i, Any, Expr(:call, x∇, ex.args...))
    ir[SSAValue(i)] = xgetindex(yJ, 1)
    insert_node!(ir, i+1, Any, xgetindex(yJ, 2), true)
  end
  ir, map = _compact!(ir)
  return ir, rename(xs, map)
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

struct ReverseIR
  forw::IRCode
  perm::Vector{Int}
  stmts::Vector{Any}
  blocks::Vector{BasicBlock}
end

ReverseIR(ir::IRCode) = ReverseIR(ir, reverse_order(ir.cfg), [], [])

Base.push!(ir::ReverseIR, x) = push!(ir.stmts, x)

function block!(ir::ReverseIR)
  start = isempty(ir.blocks) ? 1 : ir.blocks[end].stmts.last+1
  old = ir.forw.cfg.blocks[invperm(ir.perm)[length(ir.blocks)+1]]
  newidx(i) = ir.perm[i]
  preds, succs = newidx.(old.succs), newidx.(old.preds)
  if isempty(succs)
  elseif length(succs) == 1
    push!(ir, GotoNode(succs[1]))
  else
    push!(ir, GotoIfNot(Alpha(range(old)[1]), succs[1]))
    push!(ir, GotoNode(succs[2]))
  end
  push!(ir.blocks, BasicBlock(StmtRange(start,length(ir.stmts)), preds, succs))
end

IRCode(ir::ReverseIR) =
  IRCode(ir.forw, ir.stmts, Any[Any for _ in ir.stmts], [-1 for _ in ir.stmts], CFG(ir.blocks), NI.NewNode[])

function grad!(ir::ReverseIR, grads, i)
  ex = ir.forw.stmts[i]
  (isexpr(ex, :call) && ex.args[1] == x∇) || return
  Δ = grads[SSAValue(i+1)]
  J = Alpha(i+2)
  push!(ir, Expr(:call, J, Δ))
  Δ = SSAValue(length(ir.stmts))
  for (i, x) in enumerate(ex.args[3:end])
    haskey(grads, x) || continue
    push!(ir, xgetindex(Δ, i))
    push!(ir, xaccum!(grads[x], SSAValue(length(ir.stmts))))
  end
end

function reverse_ir(ir::IRCode, xs)
  ir, grads = ReverseIR(ir), Dict()
  push!(ir, :(Δ()))
  for x in xs # TODO: put these in the right block
    push!(ir, Expr(:call, GlobalRef(Zygote, :grad), alpha(x)))
    grads[x] = SSAValue(length(ir.stmts))
  end
  for (bi, b) in enumerate(ir.forw.cfg.blocks[ir.perm])
    for i in reverse(range(b))
      grad!(ir, grads, i)
    end
    bi == length(ir.forw.cfg.blocks) && push!(ir, ReturnNode(grads[Argument(2)]))
    block!(ir)
  end
  return IRCode(ir)
end

function grad_ir(ir)
  validcfg(ir) || error("Multiple return not supported")
  forw, xs = record!(record_branches!(ir))
  back = reverse_ir(forw, xs)
  return forw, back
end
