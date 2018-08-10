using Base: RefValue

accum!(r::RefValue, x) = (r.x = accum(r.x, deref(x)))

function accumif!(c::Bool, r::RefValue, x)
  c && accum!(r, x)
  return
end

deref!(x) = x

function deref!(r::RefValue)
  y = r.x
  r.x = nothing
  return y
end
iscall(x, m::Module, n::Symbol) = isexpr(x, :call) && x.args[1] == GlobalRef(m, n)

function merge_returns(ir)
  any(x -> x == unreachable, ir.stmts) && error("`throw` not supported")
  rs = findall(x -> x isa ReturnNode, ir.stmts)
  length(rs) <= 1 && return ir
  bs = blockidx.(Ref(ir), rs)
  xs = []
  bb = length(ir.cfg.blocks)+1
  @assert length(unique(bs)) == length(bs)
  for r in rs
    x = ir.stmts[r].val
    x == GlobalRef(Base, :nothing) && (x = nothing)
    @assert !(x isa Union{GlobalRef,Expr})
    push!(xs, x)
    ir.stmts[r] = GotoNode(bb)
  end
  push!(ir.cfg.blocks, BasicBlock(StmtRange(length(ir.stmts), length(ir.stmts)-1), bs, []))
  push!(ir.cfg.index, length(ir.stmts))
  for b in bs
    push!(ir.cfg.blocks[b].succs, bb)
  end
  ir = IncrementalCompact(ir)
  for _ in ir end
  # TODO preserve types
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

xaccum!(x, Δ; cond = nothing) =
  cond == nothing ?
    xcall(Zygote, :accum!, x, Δ) :
    xcall(Zygote, :accumif!, cond, x, Δ)

function record_branches!(ir::IRCode)
  ir = IncrementalCompact(ir)
  offset = 0
  for (i, x) in ir
    bi = findfirst(x -> x == i+1-offset, ir.ir.cfg.index)
    bi == nothing && continue
    preds = ir.ir.cfg.blocks[bi+1].preds
    length(preds) > 1 || continue
    @assert length(preds) <= 2
    insert_node_here!(ir, PhiNode(sort(preds), [false, true]), Bool, ir.result_lines[i])
    offset += 1
  end
  return finish_dc(ir)
end

istrackable(x) =
  x isa GlobalRef && x.mod ∉ (Base, Core) &&
  !(isconst(x.mod, x.name) && typeof(getfield(x.mod, x.name)) <: Union{Function,Type})

function record_globals!(ir::IRCode)
  for i = 1:length(ir.stmts)
    ex = ir[SSAValue(i)]
    # TODO general globalrefs
    if isexpr(ex, :call)
      for j = 1:length(ex.args)
        istrackable(ex.args[j]) || continue
        ex.args[j] = insert_node!(ir, i, Any, xcall(Zygote, :unwrap, ex.args[j]))
      end
    end
  end
  return compact!(ir)
end

ignored_f(f) = f in (GlobalRef(Base, :not_int),
                     GlobalRef(Core.Intrinsics, :not_int),
                     GlobalRef(Core, :(===)),
                     GlobalRef(Core, :apply_type),
                     GlobalRef(Core, :typeof))
ignored_f(ir, f) = ignored_f(f)
ignored_f(ir, f::SSAValue) = ignored_f(ir[f])

ignored(ir, ex) = isexpr(ex, :call) && ignored_f(ir, ex.args[1])
ignored(ir, ex::SSAValue) = ignored(ir, ir[ex])

function valid_usages(ir)
  r = Dict()
  for (x, us) in usages(ir)
    x isa Union{SSAValue,Argument} || continue
    us′ = filter(i -> !ignored(ir, i), us)
    isempty(us′) || (r[x] = us′)
  end
  return r
end

reachable(ir) = keys(valid_usages(ir))

# TODO: remove this once we don't mess with type inference
function _forward_type(Ts)
  isconcretetype(Tuple{Ts...}) || return Any
  typed_meta(Tuple{Ts...}) == nothing && return Any
  T = Core.Compiler.return_type(_forward, Tuple{Context,Ts...})
  return T == Union{} ? Any : T
end

isvalidtype(jT, yT) = jT <: Tuple && length(jT.parameters) == 2 && jT.parameters[1] <: yT

function record!(ir::IRCode)
  pushfirst!(ir.argtypes, typeof(_forward), Context)
  xs = reachable(ir)
  for i = 1:length(ir.stmts)
    ex = argmap(x -> Argument(x.n+2), ir[SSAValue(i)])
    isexpr(ex, :new) && (ex = ir[SSAValue(i)] = xcall(Zygote, :__new__, ex.args...))
    if isexpr(ex, :call) && !ignored(ir, ex)
      yT = widenconst(types(ir)[i])
      T = _forward_type(exprtype.(Ref(ir), ex.args))
      if isvalidtype(T, yT)
        yJ = insert_node!(ir, i, T, xcall(Zygote, :_forward, Argument(2), ex.args...))
        ir[SSAValue(i)] = xgetindex(yJ, 1)
        insert_node!(ir, i, T.parameters[2], xgetindex(yJ, 2), true)
      else
        yJ = insert_node!(ir, i, Any, xcall(Zygote, :_forward, Argument(2), ex.args...))
        y =  insert_node!(ir, i, Any, xgetindex(yJ, 1))
        J =  insert_node!(ir, i, Any, xgetindex(yJ, 2))
        ir[SSAValue(i)] = xcall(Zygote, :typeassert, y, yT)
      end
    else
      ir[SSAValue(i)] = ex
    end
  end
  ir, m = _compact!(ir)
  return ir, Set(x isa Argument ? Argument(x.n+2) : x for x in rename(xs, m))
end

function reverse_cfg(cfg, perm)
  newidx(i) = invperm(perm)[i]
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
  lines::Vector{Int32}
  blocks::Vector{BasicBlock}
  uses::Dict{Any,Any}
end

ReverseIR(ir::IRCode) = ReverseIR(ir, reverse_order(ir.cfg), [], [], [], usages(ir))

function Base.push!(ir::ReverseIR, x, i = 0)
  push!(ir.stmts, x)
  push!(ir.lines, i)
  return SSAValue(length(ir.stmts))
end

function block!(ir::ReverseIR)
  start = isempty(ir.blocks) ? 1 : ir.blocks[end].stmts.last+1
  old = ir.forw.cfg.blocks[ir.perm[length(ir.blocks)+1]]
  newidx(i) = invperm(ir.perm)[i]
  preds, succs = newidx.(old.succs), newidx.(sort(old.preds))
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
  IRCode(ir.forw, ir.stmts, Any[Any for _ in ir.stmts], ir.lines,
         [0x00 for _ in ir.stmts], CFG(ir.blocks), NewNode[])

function dominates(ir::ReverseIR, def, use)
  bdef, buse = blockidx.(Ref(ir.forw), (def, use))
  bdef == buse && return def.id <= use.id
  bdef, buse = invperm(ir.perm)[[bdef, buse]]
  dt = construct_domtree(reverse_cfg(ir.forw.cfg, ir.perm))
  return dominates(dt, buse, bdef)
end

dominates(ir::ReverseIR, def::Argument, use) = dominates(ir, SSAValue(1), use)

isdirect(ir::ReverseIR, x) = length(ir.uses[x]) == 1 && dominates(ir, x, ir.uses[x][1])

function xaccum_(ir::ReverseIR, grads, x, Δ; line = 0, cond = nothing)
  if isdirect(ir, x)
    ir.stmts[grads[x].id] = nothing
    grads[x] = Δ
  else
    push!(ir, xaccum!(grads[x], Δ, cond = cond), line)
  end
end

function isassert(ir, i)
  ex = ir.stmts[i+3]
  iscall(ex, Zygote, :typeassert)
end

function grad!(ir::ReverseIR, grads, i)
  ex = ir.forw.stmts[i]
  if ex isa ReturnNode && (ex.val isa SSAValue || ex.val isa Argument)
    xaccum_(ir, grads, ex.val, SSAValue(1))
  elseif ex isa PiNode
    haskey(grads, SSAValue(i)) || return
    Δ = push!(ir, xcall(Zygote, :deref!, grads[SSAValue(i)]))
    xaccum_(ir, grads, ex.val, Δ)
  elseif ex isa PhiNode
    haskey(grads, SSAValue(i)) || return
    Δ = grads[SSAValue(i)]
    @assert length(ex.edges) == 2
    rec = Alpha(range(ir.forw.cfg.blocks[blockidx(ir.forw, i)])[1])
    notrec = push!(ir, xcall(Base, :not_int, rec))
    x1, x2 = ex.values[sortperm(ex.edges)]
    haskey(grads, x1) && xaccum_(ir, grads, x1, Δ, cond = notrec)
    haskey(grads, x2) && xaccum_(ir, grads, x2, Δ, cond = rec)
  elseif iscall(ex, Zygote, :_forward)
    J = Alpha(i+2)
    line = ir.forw.lines[i]
    # TODO remove with type hacks above
    y = isassert(ir.forw, i) ? SSAValue(i+3) : SSAValue(i+1)
    Δref = get(grads, y, nothing)
    Δ = Δref == nothing ? nothing : push!(ir, xcall(Zygote, :deref!, Δref), line)
    Δ = push!(ir, Expr(:call, J, Δ), line)
    for (i, x) in enumerate(ex.args[3:end])
      haskey(grads, x) || continue
      push!(ir, xgradindex(Δ, i), line)
      xaccum_(ir, grads, x, SSAValue(length(ir.stmts)), line = line)
    end
  end
end

deref(x) = x
deref(x::RefValue) = x[]

deref_tuple(xs...) = map(deref,xs)

@inline deref_tuple_va(N, xs) = xs
@inline deref_tuple_va(N, x, xs...) = (deref(x), deref_tuple_va(N, xs...)...)
@inline deref_tuple_va(N, xs::Ref) = deref_tuple_va(N, deref(xs))
@inline deref_tuple_va(::Val{N}, ::Nothing) where N = ntuple(_ -> nothing, Val(N))

# TODO: another type hack. We should be using phis on the backward pass
gradtype(_) = Any
gradtype(T::Type{<:Real}) = float(T)
Base.convert(T::Type{<:Real}, ::Nothing) = zero(T)

function reverse_ir(forw::IRCode, xs; varargs = nothing)
  ir, grads = ReverseIR(forw), Dict()
  push!(ir, :(Δ()))
  for x in xs
    T = gradtype(exprtype(forw, x))
    push!(ir, Expr(:call, Ref{T}, nothing))
    grads[x] = SSAValue(length(ir.stmts))
  end
  for (bi, b) in enumerate(ir.forw.cfg.blocks[ir.perm])
    for i in reverse(range(b))
      grad!(ir, grads, i)
    end
    if ir.perm[bi] == 1
      gs = [get(grads, Argument(i), nothing) for i = 3:length(forw.argtypes)]
      if varargs == nothing
        push!(ir, xcall(Zygote, :deref_tuple, gs...))
      else
        push!(ir, xcall(Zygote, :deref_tuple_va, Val(varargs), gs...))
      end
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

function grad_ir(ir; varargs = nothing)
  ir = merge_returns(ir)
  forw, xs = record!(record_branches!(record_globals!(ir)))
  back, perm = reverse_ir(forw, xs, varargs = varargs)
  return Adjoint(forw, compact!(back), perm)
end

using InteractiveUtils: @which

macro adjoint(ex)
  :(grad_ir($(code_irm(ex)), varargs = varargs($(esc(:(@which $ex))), length(($(esc.(ex.args)...),)))))
end
