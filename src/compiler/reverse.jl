using Base: RefValue

deref(x) = x
deref(x::RefValue) = x[]
gradref(x) = RefValue(grad(x))

accum!(r::RefValue, x) = (r.x = accum(r.x, deref(x)))

backprop(J, Δx) = J(Δx)

function backprop(J, Δx::RefValue)
  Δy = J(Δx.x)
  Δx.x = grad(Δx.x)
  return Δy
end

function merge_returns(ir)
  rs = findall(x -> x isa ReturnNode, ir.stmts)
  length(rs) <= 1 && return ir
  bs = blockidx.(ir, rs)
  xs = []
  bb = length(ir.cfg.blocks)+1
  @assert length(unique(bs)) == length(bs)
  for r in rs
    push!(xs, ir.stmts[r].val)
    ir.stmts[r] = GotoNode(bb)
  end
  push!(ir.cfg.blocks, BasicBlock(StmtRange(length(ir.stmts), length(ir.stmts)-1), bs, []))
  push!(ir.cfg.index, length(ir.stmts))
  for b in bs
    push!(ir.cfg.blocks[b].succs, bb)
  end
  ir = IncrementalCompact(ir)
  for _ in ir end
  r = insert_node_here!(ir, PhiNode(bs, xs), Any, ir.result_lines[end])
  insert_node_here!(ir, ReturnNode(r), Any, ir.result_lines[end])
  ir = finish(ir)
  return ir
end

struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::SSAValue) = Alpha(x.id)

gradindex(x, i) = x[i]
gradindex(::Nothing, i) = nothing
xgetindex(x, i...) = Expr(:call, GlobalRef(Base, :getindex), x, i...)
xgradindex(x, i) = xcall(Zygote, :gradindex, x, i)

xaccum!(x, Δ) = Expr(:call, GlobalRef(Zygote, :accum!), x, Δ)

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
  return finish_dc(ir)
end

function reachable(ir)
  seen = []
  stack = [ir.stmts[end].val]
  while !isempty(stack)
    i = popfirst!(stack)
    i ∈ seen && continue
    push!(seen, i)
    i isa SSAValue || continue
    for x in userefs(ir[i])
      x[] isa SSAValue && push!(stack, x[])
      x[] isa Argument && x[] ∉ seen && push!(seen, x[])
    end
  end
  return seen
end

ignored(f) = f in (GlobalRef(Base, :not_int), GlobalRef(Core.Intrinsics, :not_int), GlobalRef(Core, :(===)))
ignored(ir, f) = ignored(f)
ignored(ir, f::SSAValue) = ignored(ir[f])

function record!(ir::IRCode)
  xs = reachable(ir)
  for i = 1:length(ir.stmts)
    ex = argmap(x -> Argument(x.n+2), ir[SSAValue(i)])
    isexpr(ex, :new) && (ex = ir[SSAValue(i)] = xcall(Zygote, :__new__, ex.args...))
    if isexpr(ex, :call)
      ignored(ir, ex.args[1]) && continue
      yJ = insert_node!(ir, i, Any, xcall(Zygote, :_forward, Argument(2), ex.args...))
      ir[SSAValue(i)] = xgetindex(yJ, 1)
      insert_node!(ir, i, Any, xgetindex(yJ, 2), true)
    else
      ir[SSAValue(i)] = ex
    end
  end
  ir, m = _compact!(ir)
  return ir, map(x -> x isa Argument ? Argument(x.n+2) : x, rename(xs, m))
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
  uses::Dict{Any,Any}
end

ReverseIR(ir::IRCode) = ReverseIR(ir, reverse_order(ir.cfg), [], [], usages(ir))

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
  IRCode(ir.forw, ir.stmts, Any[Any for _ in ir.stmts], [1 for _ in ir.stmts],
         [0x00 for _ in ir.stmts], CFG(ir.blocks), NewNode[])

function dominates(ir::ReverseIR, def, use)
  bdef, buse = blockidx.(ir.forw, (def, use))
  bdef == buse && return def.id <= use.id
  bdef, buse = ir.perm[[bdef, buse]]
  dt = construct_domtree(reverse_cfg(ir.forw.cfg, ir.perm))
  return dominates(dt, buse, bdef)
end

dominates(ir::ReverseIR, def::Argument, use) = dominates(ir, SSAValue(1), use)

# TODO don't have special semantics here
function xaccum_(ir::ReverseIR, grads, x, Δ)
  if length(ir.uses[x]) == 1 && dominates(ir, x, ir.uses[x][1])
    ir.stmts[grads[x].id] = nothing
    grads[x] = Δ
  else
    push!(ir, xaccum!(grads[x], Δ))
  end
end

function phis!(ir::ReverseIR, grads, bi)
  succs = ir.forw.cfg.blocks[bi].succs
  for s in succs
    for i in range(ir.forw.cfg.blocks[s])
      (ex = ir.forw.stmts[i]) isa PhiNode || break
      haskey(grads, SSAValue(i)) || continue
      x = ex.values[findfirst(e -> e == bi, ex.edges)]
      haskey(grads, x) || continue
      @assert length(succs) == 1
      xaccum_(ir, grads, x, grads[SSAValue(i)])
    end
  end
end

function grad!(ir::ReverseIR, grads, i)
  ex = ir.forw.stmts[i]
  if ex isa ReturnNode && (ex.val isa SSAValue || ex.val isa Argument)
    xaccum_(ir, grads, ex.val, SSAValue(1))
  elseif isexpr(ex, :call) && ex.args[1] == GlobalRef(Zygote, :_forward)
    Δ = get(grads, SSAValue(i+1), nothing)
    J = Alpha(i+2)
    push!(ir, Expr(:call, GlobalRef(Zygote, :backprop), J, Δ))
    Δ = SSAValue(length(ir.stmts))
    for (i, x) in enumerate(ex.args[3:end])
      haskey(grads, x) || continue
      push!(ir, xgradindex(Δ, i))
      xaccum_(ir, grads, x, SSAValue(length(ir.stmts)))
    end
  end
end

deref_tuple(xs...) = deref.(xs)
@inline deref_tuple_va(xs) = deref(xs)
@inline deref_tuple_va(x, xs...) = (deref(x), deref_tuple_va(xs...)...)

function reverse_ir(forw::IRCode, xs; varargs = false)
  ir, grads = ReverseIR(forw), Dict()
  push!(ir, :(Δ()))
  for x in xs # TODO: put these in the right block
    push!(ir, Expr(:call, GlobalRef(Zygote, :gradref), alpha(x)))
    grads[x] = SSAValue(length(ir.stmts))
  end
  for (bi, b) in enumerate(ir.forw.cfg.blocks[ir.perm])
    phis!(ir, grads, invperm(ir.perm)[bi])
    for i in reverse(range(b))
      grad!(ir, grads, i)
    end
    if bi == length(ir.forw.cfg.blocks)
      gs = [get(grads, Argument(i+2), nothing) for i = 1:length(forw.argtypes)]
      push!(ir, xcall(Zygote, varargs ? :deref_tuple_va : :deref_tuple, gs...))
      push!(ir, ReturnNode(SSAValue(length(ir.stmts))))
    end
    block!(ir)
  end
  return IRCode(ir), ir.perm
end

struct Adjoint
  forw::IRCode
  back::IRCode
  perm::Vector{Int}
end

function grad_ir(ir; varargs = false)
  ir = merge_returns(ir)
  forw, xs = record!(record_branches!(ir))
  back, perm = reverse_ir(forw, xs, varargs = varargs)
  return Adjoint(forw, compact!(back), perm)
end

macro code_grad(ex)
  # TODO fix escaping
  :(grad_ir($(code_irm(ex)), varargs = $(esc(:(@which $ex))).isva))
end
