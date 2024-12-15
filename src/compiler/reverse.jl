using IRTools: IR, Variable, Pipe, xcall, var, prewalk, postwalk,
  blocks, predecessors, successors, argument!, arguments, branches,
  insertafter!, finish, expand!, prune!, substitute!, substitute,
  block, block!, branch!, return!, stmt, meta


# TODO: Temporary, to be removed when ChainRulesCore rrules are required to
# support thunks as an input and all instances of _adjoint_keepthunks in
# Zygote have been replaces by rrules:
macro _adjoint_keepthunks(ex)
  ZygoteRules.gradm(ex, false, true)
end
macro _adjoint_keepthunks!(ex)
  ZygoteRules.gradm(ex, true, true)
end


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

is_getproperty(ex) = iscall(ex, Base, :getproperty)

# The initial premise of literal_getproperty was in some ways inherently flawed, because for
# getproperty it was intended that _pullback falls back to literal_getproperty, but we actually
# want the opposite to happen, since Zygote should fall back to recursing into the getproperty
# implementation by default. Users still want to define custom adjoints using only
# literal_getproperty, though. We can't really have mutually recursive definitions here, so we
# now always instrument getproperty as literal_getproperty, no matter whether the second
# argument is a literal or not.
function instrument_getproperty!(ir, v, ex)
  if is_getproperty(ex)
    obj, prop = ex.args[2], ex.args[3]
    if obj isa Module && prop isa QuoteNode && isconst(obj, unwrapquote(prop))
      # Metaprogramming can generate getproperty(::Module, ...) calls.
      # Like other types, these are type unstable without constprop.
      # However, literal_getproperty's heuristic is also not general enough for modules.
      # Thankfully, we can skip instrumenting these if they're const properties.
      ex
    elseif prop isa Union{QuoteNode,Integer}
      ir[v] = xcall(Zygote, :literal_getproperty, obj, Val(unwrapquote(prop)))
    else
      f = insert!(ir, v, :(Val($(prop))))
      ir[v] = xcall(Zygote, :literal_getproperty, obj, f)
    end
  else
    ex
  end
end

is_literal_getfield(ex) =
  (iscall(ex, Core, :getfield) || iscall(ex, Base, :getfield)) &&
  ex.args[3] isa Union{QuoteNode,Integer}

# Here, only instrumenting getfield with literals is fine, since users should never have to
# define custom adjoints for literal_getfield
function instrument_getfield!(ir, v, ex)
  if is_literal_getfield(ex)
    ir[v] = xcall(Zygote, :literal_getfield, ex.args[2], Val(unwrapquote(ex.args[3])))
  else
    ex
  end
end

is_literal_getindex(ex) =
  iscall(ex, Base, :getindex) && length(ex.args) == 3 && ex.args[3] isa Union{Integer,QuoteNode}

# TODO: is this always correct for user defined getindex methods?
function instrument_getindex!(ir, v, ex)
  if is_literal_getindex(ex)
    ir[v] = xcall(Zygote, :literal_getindex, ex.args[2], Val(unwrapquote(ex.args[3])))
  else
    ex
  end
end

is_literal_iterate(ex) =
  iscall(ex, Base, :indexed_iterate) && length(ex.args) >= 3 && ex.args[3] isa Union{Integer,QuoteNode}

function instrument_iterate!(ir, v, ex)
  if is_literal_iterate(ex)
    ir[v] = xcall(Zygote, :literal_indexed_iterate, ex.args[2],
                  Val(unwrapquote(ex.args[3])), ex.args[4:end]...)
  else
    ex
  end
end

function instrument_literals!(ir, v, ex)
  ex = instrument_getproperty!(ir, v, ex)
  ex = instrument_getfield!(ir, v, ex)
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
    if isexpr(ex, :foreigncall, :isdefined)
      continue
    elseif isexpr(ex, :(=))
      @assert ex.args[1] isa GlobalRef
      pr[v] = xcall(Zygote, :global_set, QuoteNode(ex.args[1]), ex.args[2])
    elseif isexpr(ex, :boundscheck)
      # Expr(:boundscheck) now appears in common Julia code paths, so we need to handle it.
      # For correctness sake, fix to true like https://github.com/dfdx/Umlaut.jl/issues/34.
      pr[v] = true
    else
      ex = instrument_new!(pr, v, ex)
      ex = instrument_literals!(pr, v, ex)
      ex = instrument_global!(pr, v, ex)
    end
  end
  ir = finish(pr)
  # GlobalRefs can turn up in branch arguments
  for b in blocks(ir), br in branches(b), i in 1:length(arguments(br))
    (ref = arguments(br)[i]) isa GlobalRef || continue
    arguments(br)[i] = push!(b, xcall(Zygote, :unwrap, QuoteNode(ref), ref))
  end
  return ir
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

function ignored(ir, ex)
  isexpr(ex, :call) || return false
  f = ex.args[1]
  ignored_f(ir, f) && return true
  if f isa Variable && haskey(ir, f)
    f = ir[f].expr
  end
  if f == GlobalRef(Base, :getproperty) && length(ex.args) >= 3
    obj, prop = ex.args[2], ex.args[3]
    # Metaprogramming can generate getproperty(::Module, ...) calls.
    # These are type unstable without constprop, which transforming to _pullback breaks.
    # However, we can skip differentiating these if they're const properties.
    obj isa Module && prop isa QuoteNode && isconst(obj, unwrapquote(prop)) && return true
  end
  return false
end
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
    if isempty(preds) || (!isempty(branches(b)) && branches(b)[end] == IRTools.unreachable)
      # If `b` is unreachable, then no context produced by the primal should end up branching to `rb`
      push!(rb, xcall(Base, :error, "unreachable")) # `throw` is necessary for inference not to hit the `unreachable`
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

function passthrough_expr(ex::Expr)
    # Metadata we want to preserve
    isexpr(ex, GlobalRef, :call, :isdefined, :inbounds, :meta, :loopinfo, :enter, :leave, :catch) && return true
    # ccalls and more that are safe to preserve/required for proper operation:
    # - jl_set_task_threadpoolid: added in 1.9 for @spawn
    isexpr(ex, :foreigncall) && unwrapquote(ex.args[1]) in (:jl_set_task_threadpoolid,) && return true
    return false
end

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

    has_leave = false

    # Backprop through statements
    for v in reverse(keys(b))
      ex = b[v].expr
      has_leave |= isexpr(ex, :leave)

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
      elseif isexpr(ex) && !passthrough_expr(ex)
        push!(rb, stmt(xcall(Base, :error, """
                             Can't differentiate $(ex.head) expression $ex.
                             You might want to check the Zygote limitations documentation.
                             https://fluxml.ai/Zygote.jl/latest/limitations
                             """),
                       line = b[v].line))
      else # A literal value
        continue
      end
    end

    # This is corresponds to a catch blocks which technically
    # has predecessors but they are not modelled in the IRTools CFG.
    # We put an error message at the beginning of said block.
    if has_leave && isempty(predecessors(b)) && b.id != 1
        _, f_stmt = first(b)
        li = pr.ir.lines[f_stmt.line]
        pushfirst!(rb, stmt(xcall(Base, :error,
                                  "Can't differentiate function execution in catch block at $(li.file):$(li.line).")))
    end

    if b.id > 1 # Backprop through (predecessor) branch arguments
      gs = grad.(arguments(b))
      for br in branches(rb)
        br.block == 0 && continue
        br′ = branchfor(pr.ir, br.block=>b.id)
        br′ === nothing && continue
        ins = br′.args
        for i = 1:length(br.args)
          ā = [gs[j] for j = 1:length(ins) if ins[j] == sigs[br.block][i]]
          br.args[i] = xaccum(rb, ā...)
        end
      end
    else # Backprop function arguments
      gs = [grad(arg) for arg = arguments(pr.ir)]
      Δ = push!(rb, pr.varargs === nothing ?
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
