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

function _push!(a::Vector{T}, x::T) where T
  Base._growend!(a, 1)
  @inbounds a[end] = x
  return
end

# Grows a vector typed after the first value pushed, so that values aren't boxed; the backward
# pass sees the concrete type through `typed_pullback`.
mutable struct StackRef
  v::Any
end

StackRef() = StackRef(nothing)

function _push!(s::StackRef, x::T) where T
  v = s.v
  if v isa Vector{T}
    _push!(v, x)
  elseif v === nothing
    s.v = T[x]
  elseif x isa eltype(v)
    push!(v, x)
  else
    s.v = push!(Vector{Any}(v), x)
  end
  return
end

unstack(x) = x
unstack(s::StackRef) = s.v === nothing ? Any[] : s.v

# Emit

xstack(T) = Expr(:call, Vector{T})

function alphauses(b)
  us = Set{Alpha}()
  postwalk(x -> x isa Alpha && push!(us, x), b)
  return us
end

xtuple(xs...) = xcall(:tuple, xs...)

concrete(T::DataType) = T
concrete(::Type{Type{T}}) where T = typeof(T)
concrete(T) = Any

# a cycle needs a branch to an earlier (or the same) block
hasloops(ir) = any(b -> any(br -> 0 < br.block <= b.id, branches(b)), blocks(ir))

runonce(b) = b.id in (1, length(b.ir.blocks)) &&
             !any(((_,stmt),) -> isexpr(stmt.expr, :catch), b)

# Each block of acyclic code runs at most once, so the values it records can reach the last block
# as block arguments (`threaded!`) instead of going through stacks. The record then holds `nothing`
# for the blocks that didn't run, and `_pullback` types it with its inferred type (#1754). Not with
# `try`, as a `catch` block is entered without a branch.
typed_branches(ir) = length(blocks(ir)) > 1 && !hasloops(ir) &&
                     !any(((_,st),) -> isexpr(st.expr, :enter, :catch), ir)

# `x`, defined in block `b`, as seen at the start of the last block: `nothing` on the paths that
# skip `b`. Blocks are in topological order, since no branch goes back.
function threaded!(ir, preds, b, x)
  out = Vector{Any}(undef, length(blocks(ir)))
  local x′
  for blk in blocks(ir)
    xs = [out[p] for p in preds[blk.id]]
    if isempty(xs)
      x′ = nothing
    elseif allequal(xs)
      x′ = xs[1]
    else
      x′ = argument!(blk, nothing)
      for p in preds[blk.id], br in branches(block(ir, p))
        br.block == blk.id && (arguments(br)[end] = out[p])
      end
    end
    out[blk.id] = blk.id == b ? x : x′
  end
  return x′
end

function forward_stacks!(adj, F)
  stks, recs = [], []
  pr = adj.primal
  loopy = hasloops(pr)
  typed = typed_branches(pr)
  preds = predecessor_ids(pr)
  for b in blocks(pr)
    not_stack = runonce(b)
    for α in alphauses(block(adj.adjoint, b.id))
      if not_stack
        push!(recs, Variable(α))
      elseif typed
        push!(recs, threaded!(pr, preds, b.id, Variable(α)))
      else
        stk = pushfirst!(pr, loopy ? xcall(Zygote, :StackRef) : xstack(Any))
        push!(recs, stk)
        push!(b, xcall(Zygote, :_push!, stk, Variable(α)))
      end
      push!(stks, (b.id, alpha(α), typed ? :slot : not_stack ? :value : :stack))
    end
  end
  rec = push!(pr, xtuple(recs...))
  rec = length(pr.blocks) == 1 ? push!(pr, Expr(:call, Pullback{F}, rec)) :
        typed ? rec : push!(pr, xcall(Zygote, :branchy_pullback, F, rec, Val(loopy)))
  ret = xtuple(pr.blocks[end].branches[end].args[1], rec)
  ret = push!(pr, ret)
  pr.blocks[end].branches[end].args[1] = ret
  return pr, stks
end

# The forward pass of `typed_branches` code returns its record untyped, which `_pullback_record`
# keeps. `_pullback` types it after the type inferred for `_pullback_record`, a constant.
function typed_record!(forw, T)
  b = blocks(forw)[end]
  ret = forw[branches(b)[end].args[1]].expr
  y, rec = ret.args[2], ret.args[3]
  cx = push!(b, xcall(Core, :typeof, arguments(forw)[2]))
  sig = push!(b, xcall(Core, :apply_type, Tuple, cx, T.parameters...))
  R = push!(b, xcall(Core.Compiler, :return_type, GlobalRef(Zygote, :_pullback_record), sig))
  xs = [push!(b, xcall(Zygote, :slot, R, Val(i), x)) for (i, x) in enumerate(forw[rec].expr.args[2:end])]
  pb = push!(b, Expr(:call, Pullback{T}, push!(b, xtuple(xs...))))
  branches(b)[end].args[1] = push!(b, xtuple(y, pb))
  return forw
end

function reverse_stacks!(adj, stks)
  ir = adj.adjoint
  entry = block(ir, length(blocks(adj.primal)))
  self = argument!(entry, at = 1)
  t = pushfirst!(entry, xcall(:getfield, self, QuoteNode(:t)))
  repl = Dict()
  for (i, (b′, α, kind)) in enumerate(stks)
    if kind === :stack
      stk = push!(entry, xcall(:getindex, t, i))
      stk = push!(entry, xcall(Zygote, :Stack, stk))
      val = pushfirst!(block(ir, b′), xcall(:pop!, stk))
    else
      val = insertafter!(ir, t, xcall(:getindex, t, i))
      kind === :slot && (val = insertafter!(ir, val, xcall(Zygote, :unslot, val)))
    end
    repl[α] = val
  end
  return IRTools.prewalk!(x -> get(repl, x, x), ir)
end

function stacks!(adj, T)
  forw, stks = forward_stacks!(adj, T)
  back = reverse_stacks!(adj, stks)
  permute!(back, reverse_order(back, length(blocks(adj.primal))))
  IRTools.domorder!(back)
  return forw, back
end

varargs(m::Method, n) = m.isva ? n - m.nargs + 1 : nothing

# The generators decompose the same signature many times in a row (the forward and the backward
# pass, for each context and cotangent type), so the result is kept until the world age moves.
# an `IdDict`, since a `Dict` would be recompiled for the type of every signature used as key
const decompositions = IdDict{Any,Tuple{IR,IR}}()
const decompositions_world = Ref(UInt(0))
const decompositions_lock = ReentrantLock()

function clear_decompositions!()
  @lock decompositions_lock begin
    empty!(decompositions)
    decompositions_world[] = 0
  end
end

function cached_decomposition(T, world)
  @lock decompositions_lock begin
    decompositions_world[] == world ? get(decompositions, T, nothing) : nothing
  end
end

function cache_decomposition!(T, world, forw, back)
  # drop the metadata, which holds the `CodeInfo` that the generators go on to rewrite
  entry = (IR(forw.defs, forw.blocks, forw.lines, nothing),
           IR(back.defs, back.blocks, back.lines, nothing))
  @lock decompositions_lock begin
    if world > decompositions_world[]
      empty!(decompositions)
      decompositions_world[] = world
    end
    # a generator for an older world can't reuse or replace what newer ones use
    world == decompositions_world[] && (decompositions[T] = entry)
  end
  return
end

# `copy(::IR)` shares the expressions, which the passes after `stacks!` rewrite in place.
copy_ir(ir::IR, meta) = (ir = copy(IRTools.prewalk(identity, ir)); IR(ir.defs, ir.blocks, ir.lines, meta))

# `meta` and one of the two pieces of IR (`:forward` or `:backward`) for `T`, to be modified freely.
function _generate_pullback_via_decomposition(T, world, side::Symbol)
  (m = meta(T; world)) === nothing && return
  entry = cached_decomposition(T, world)
  if entry === nothing
    va = varargs(m.method, length(T.parameters))
    forw, back = stacks!(Adjoint(IR(m); varargs = va, normalise = false, world), T)
    cache_decomposition!(T, world, forw, back)
    entry = (forw, back)
  end
  return m, copy_ir(side === :forward ? entry[1] : entry[2], m)
end

function stacklines(T::Type)
  adj = Adjoint(IR(meta(T)), normalise = false)
  recs = []
  for b in blocks(adj.adjoint), α in alphauses(b)
    push!(recs, IRTools.exprline(adj.primal, Variable(α)))
  end
  return recs
end
