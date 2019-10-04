using IRTools: IR, Variable, Pipe, xcall, var, prewalk, postwalk,
  blocks, predecessors, successors, argument!, arguments, branches,
  insertafter!, finish, expand!, prune!, substitute!, substitute,
  block, block!, branch!, return!, stmt, meta
using Base: @get!

@inline tuple_va(N, xs) = xs
@inline tuple_va(N, x, xs...) = (x, tuple_va(N, xs...)...)
@inline tuple_va(::Val{N}, ::Nothing) where N = ntuple(_ -> nothing, Val(N))

iscall(x, m::Module, n::Symbol) = isexpr(x, :call) && x.args[1] == GlobalRef(m, n)

gradindex(x, i) = x[i]
gradindex(::Nothing, i) = nothing
xgetindex(x, i...) = xcall(Base, :getindex, x, i...)
xgradindex(x, i) = xcall(Zygote, :gradindex, x, i)

normalise!(ir) = ir |> IRTools.merge_returns!

function instrument_new!(ir, v, ex)
  isexpr(ex, :new) ? (ir[v] = xcall(Zygote, :__new__, ex.args...)) :
  isexpr(ex, :splatnew) ? (ir[v] = xcall(Zygote, :__splatnew__, ex.args...)) :
  ex
end

# Hack to work around fragile constant prop through overloaded functions
unwrapquote(x) = x
unwrapquote(x::QuoteNode) = x.value

is_literal_getproperty(ex) =
  (iscall(ex, Base, :getproperty) || iscall(ex, Core, :getfield) || iscall(ex, Base, :getfield)) &&
  ex.args[3] isa Union{QuoteNode,Integer}

function instrument_getproperty!(ir, v, ex)
  is_literal_getproperty(ex) ?
    (ir[v] = xcall(Zygote, :literal_getproperty, ex.args[2], Val(unwrapquote(ex.args[3])))) :
    ex
end

is_literal_getindex(ex) =
  iscall(ex, Base, :getindex) && length(ex.args) == 3 && ex.args[3] isa Union{Integer,QuoteNode}

function instrument_getindex!(ir, v, ex)
  is_literal_getindex(ex) ?
    (ir[v] = xcall(Zygote, :literal_getindex, ex.args[2], Val(unwrapquote(ex.args[3])))) :
    ex
end

is_literal_iterate(ex) =
  iscall(ex, Base, :indexed_iterate) && length(ex.args) >= 3 && ex.args[3] isa Union{Integer,QuoteNode}

function instrument_iterate!(ir, v, ex)
  is_literal_iterate(ex) ?
    (ir[v] = xcall(Zygote, :literal_indexed_iterate, ex.args[2],
                   Val(unwrapquote(ex.args[3])), ex.args[4:end]...)) :
    ex
end

function instrument_literals!(ir, v, ex)
  ex = instrument_getproperty!(ir, v, ex)
  ex = instrument_getindex!(ir, v, ex)
  ex = instrument_iterate!(ir, v, ex)
end

function istrackable(x)
  x isa GlobalRef && x.mod ∉ (Base, Core) || return false
  isconst(x.mod, x.name) || return true
  x = getfield(x.mod, x.name)
  !(x isa Type || sizeof(x) == 0)
end

function instrument_global!(ir, v, ex)
  if istrackable(ex)
    ir[v] = xcall(Zygote, :unwrap, QuoteNode(ex), ex)
  else
    ir[v] = prewalk(ex) do x
      istrackable(x) || return x
      insert!(ir, v, xcall(Zygote, :unwrap, QuoteNode(x), x))
    end
  end
end

function instrument(ir::IR)
  pr = Pipe(ir)
  for (v, st) in pr
    ex = st.expr
    isexpr(ex, :foreigncall, :isdefined) && continue
    isexpr(ex, :enter, :leave) && error("try/catch is not supported.")
    ex = instrument_new!(pr, v, ex)
    ex = instrument_literals!(pr, v, ex)
    ex = instrument_global!(pr, v, ex)
  end
  return finish(pr)
end

const BranchNumber = UInt8

function record_branches!(ir::IR)
  brs = Dict{Int,Variable}()
  for bb in blocks(ir)
    preds = predecessors(bb)
    length(preds) > 1 || continue
    brs[bb.id] = argument!(bb, BranchNumber(0), BranchNumber)
    i = length(arguments(bb))
    n = 0
    for aa in blocks(ir), br in branches(aa)
      br.block == bb.id && (arguments(br)[i] = BranchNumber(n += 1))
    end
  end
  return ir, brs
end

ignored_f(f) = f in (GlobalRef(Base, :not_int),
                     GlobalRef(Core.Intrinsics, :not_int),
                     GlobalRef(Core, :(===)),
                     GlobalRef(Core, :apply_type),
                     GlobalRef(Core, :typeof),
                     GlobalRef(Core, :throw),
                     GlobalRef(Base, :kwerr),
                     GlobalRef(Core, :kwfunc),
                     GlobalRef(Core, :isdefined))
ignored_f(ir, f) = ignored_f(f)
ignored_f(ir, f::Variable) = ignored_f(get(ir, f, nothing))

ignored(ir, ex) = isexpr(ex, :call) && ignored_f(ir, ex.args[1])
ignored(ir, ex::Variable) = ignored(ir, ir[ex])

function primal(ir::IR)
  pr = Pipe(ir)
  pbs = Dict{Variable,Variable}()
  argument!(pr, at = 1)
  cx = argument!(pr, Context, at = 2)
  for (v, st) in pr
    ex = st.expr
    if isexpr(ex, :call) && !ignored(ir, ex)
      yJ = insert!(pr, v, stmt(xcall(Zygote, :_pullback, cx, ex.args...),
                               line = ir[v].line))
      pr[v] = xgetindex(yJ, 1)
      J = insertafter!(pr, v, stmt(xgetindex(yJ, 2),
                                   line = ir[v].line))
      pbs[v] = substitute(pr, J)
    end
  end
  pr = finish(pr)
  pr, brs = record_branches!(pr)
  return pr, brs, pbs
end

struct Primal
  ir::IR
  pr::IR
  varargs::Union{Int,Nothing}
  branches::Dict{Int,Variable}
  pullbacks::Dict{Variable,Variable}
end

function Primal(ir::IR; varargs = nothing)
  ir = instrument(normalise!(ir))
  pr, brs, pbs = primal(ir)
  Primal(expand!(ir), pr, varargs, brs, pbs)
end

# Backwards Pass

struct Alpha
  id::Int
end

Base.show(io::IO, x::Alpha) = print(io, "@", x.id)

alpha(x) = x
alpha(x::Variable) = Alpha(x.id)
Variable(a::Alpha) = Variable(a.id)

sig(b::IRTools.Block) = unique([arg for br in branches(b) for arg in br.args if arg isa Variable])
sig(pr::Primal) = Dict(b.id => sig(b) for b in blocks(pr.ir))

# TODO unreachables?
function adjointcfg(pr::Primal)
  ir = empty(pr.ir)
  return!(ir, nothing)
  for b in blocks(pr.ir)[2:end]
    block!(ir)
    preds = predecessors(b)
    rb = block(ir, b.id)
    for i = 1:length(preds)
      cond = i == length(preds) ? nothing :
        push!(rb, xcall(Base, :(!==), alpha(pr.branches[b.id]), BranchNumber(i)))
      branch!(rb, preds[i].id, unless = cond)
    end
    if !isempty(branches(b)) && branches(b)[end] == IRTools.unreachable
      branch!(rb, 0)
    end
  end
  sigs = sig(pr)
  for b in blocks(ir)[1:end-1], i = 1:length(sigs[b.id])
    argument!(b)
  end
  argument!(blocks(ir)[end])
  return ir, sigs
end

branchfor(ir, (from,to)) =
  get(filter(br -> br.block == to, branches(block(ir, from))), 1, nothing)

xaccum(ir) = nothing
xaccum(ir, x) = x
xaccum(ir, xs...) = push!(ir, xcall(Zygote, :accum, xs...))

function adjoint(pr::Primal)
  ir, sigs = adjointcfg(pr)
  for b in reverse(blocks(pr.ir))
    rb = block(ir, b.id)
    grads = Dict()
    grad(x, x̄) = push!(get!(grads, x, []), x̄)
    grad(x) = xaccum(rb, get(grads, x, [])...)
    # Backprop through (successor) branch arguments
    for i = 1:length(sigs[b.id])
      grad(sigs[b.id][i], arguments(rb)[i])
    end
    # Backprop through statements
    for v in reverse(keys(b))
      ex = b[v].expr
      if haskey(pr.pullbacks, v)
        g = push!(rb, stmt(Expr(:call, alpha(pr.pullbacks[v]), grad(v)),
                           line = b[v].line))
        for (i, x) in enumerate(ex.args)
          x isa Variable || continue
          grad(x, push!(rb, stmt(xgradindex(g, i),
                                 line = b[v].line)))
        end
      elseif ex isa Core.PiNode
        grads[ex.val] = grads[v]
      elseif isexpr(ex, GlobalRef, :call, :isdefined, :inbounds, :meta)
      elseif isexpr(ex)
        push!(rb, stmt(xcall(Base, :error, "Can't differentiate $(ex.head) expression"),
                       line = b[v].line))
      else # A literal value
        continue
      end
    end
    if b.id > 1 # Backprop through (predecessor) branch arguments
      gs = grad.(arguments(b))
      for br in branches(rb)
        br.block == 0 && continue
        br′ = branchfor(pr.ir, br.block=>b.id)
        br′ == nothing && continue
        ins = br′.args
        for i = 1:length(br.args)
          ā = [gs[j] for j = 1:length(ins) if ins[j] == sigs[br.block][i]]
          br.args[i] = xaccum(rb, ā...)
        end
      end
    else # Backprop function arguments
      gs = [grad(arg) for arg = arguments(pr.ir)]
      Δ = push!(rb, pr.varargs == nothing ?
                      xcall(Zygote, :tuple, gs...) :
                      xcall(Zygote, :tuple_va, Val(pr.varargs), gs...))
      branches(rb)[1].args[1] = Δ
    end
  end
  return ir
end

struct Adjoint
  primal::IR
  adjoint::IR
end

function Adjoint(ir::IR; varargs = nothing, normalise = true)
  pr = Primal(ir, varargs = varargs)
  adj = adjoint(pr) |> prune!
  if normalise
    permute!(adj, length(adj.blocks):-1:1)
    adj = IRTools.domorder!(adj) |> IRTools.renumber
  end
  Adjoint(pr.pr, adj)
end
