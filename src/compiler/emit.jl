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
concrete(@nospecialize _) = Any

runonce(b) = b.id in (1, length(b.ir.blocks))

# TODO use a more efficient algorithm such as Johnson (1975)
# https://epubs.siam.org/doi/abs/10.1137/0204007
self_reaching(cfg, bid, visited = BitSet()) = reaches(cfg, bid, bid, visited)
function reaches(cfg, from, to, visited)
  for succ in cfg[from]
    if succ === to
      return true
    elseif succ ∉ visited
      push!(visited, succ)
      if reaches(cfg, succ, to, visited)
        return true
      end
    end
  end
  return false
end

function forward_stacks!(adj, F)
  stks, recs = Tuple{Int, Alpha, Bool}[], Variable[]
  pr = adj.primal
  blks = blocks(pr)
  last_block = length(blks)
  cfg = IRTools.CFG(pr)
  cfgᵀ = cfg'
  doms = IRTools.dominators(cfg)

  reaching_visited = BitSet()
  in_loop = map(1:last_block) do b
    empty!(reaching_visited)
    self_reaching(cfg, b, reaching_visited)
  end
  alphavars = Dict{Alpha, Variable}()
  alpha_blocks = [α => b.id for b in blks for α in alphauses(block(adj.adjoint, b.id))]
  for b in Iterators.reverse(blks)
    filter!(alpha_blocks) do (α, bid)
      if b.id in doms[bid]
        # If a block dominates this block, α is guaranteed to be present here
        αvar = Variable(α)
        for br in branches(b)
          map!(a -> a === α ? αvar : a, br.args, br.args)
        end
        push!(recs, b.id === last_block ? αvar : alphavars[α])
        push!(stks, (bid, α, false))
      elseif in_loop[bid]
        # This block is in a loop, so we're forced to insert stacks
        # Note: all alphas in loops will have stacks after the first iteration
        stk = pushfirst!(pr, xstack(Any))
        push!(recs, stk)
        push!(block(pr, bid), xcall(Zygote, :_push!, stk, Variable(α)))
        push!(stks, (bid, α, true))
      else
        # Fallback case, propagate alpha back through the CFG
        argvar = nothing
        if b.id > 1
          # Need to make sure all predecessors have a branch to add arguments to
          IRTools.explicitbranch!(b)
          argvar = argument!(b, insert=false)
        end
        if b.id === last_block
          # This alpha has been threaded all the way through to the exit block
          alphavars[α] = argvar
        end
        for br in branches(b)
          map!(a -> a === α ? argvar : a, br.args, br.args)
        end
        for pred in cfgᵀ[b.id]
          pred >= b.id && continue # TODO is this needed?
          pred_branches = branches(block(pr, pred))
          idx = findfirst(br -> br.block === b.id, pred_branches)
          if idx === nothing
            throw(error("Predecessor $pred of block $(b.id) has no branch to $(b.id)"))
          end
          branch_here = pred_branches[idx]
          push!(branch_here.args, α)
        end
        # We're not done with this alpha yet, revisit in predecessors
        return true
      end
      return false
    end
    # Prune any alphas that don't exist on this path through the CFG
    for br in branches(b)
      map!(a -> a isa Alpha ? nothing : a, br.args, br.args)
    end
  end
  @assert isempty(alpha_blocks)

  rec = push!(pr, xtuple(recs...))
  # Pullback{F,Any} reduces specialisation
  P = length(pr.blocks) == 1 ? Pullback{F} : Pullback{F,Any}
  rec = push!(pr, Expr(:call, P, rec))
  ret = xtuple(pr.blocks[end].branches[end].args[1], rec)
  ret = push!(pr, ret)
  pr.blocks[end].branches[end].args[1] = ret
  return pr, stks
end

# Helps constrain pullback function type in the backwards pass
# If we had the type, we could make this a PiNode
notnothing(::Nothing) = error()
notnothing(x) = x

function reverse_stacks!(adj, stks)
  ir = adj.adjoint
  blcks = blocks(ir)
  entry = blcks[end]
  self = argument!(entry, at = 1)
  t = pushfirst!(entry, xcall(:getfield, self, QuoteNode(:t)))
  repl = Dict{Alpha,Variable}()
  for b in blcks
    for (i, (b′, α, use_stack)) in enumerate(stks)
      b.id == b′ || continue
      # i.e. recs[i] from forward_stacks!
      val = insertafter!(ir, t, xcall(:getindex, t, i))
      if use_stack
        stk = push!(entry, xcall(Zygote, :Stack, val))
        val = pushfirst!(b, xcall(:pop!, stk))
      elseif !runonce(b)
        # The first and last blocks always run, so this check is redundant there
        val = pushfirst!(b, xcall(Zygote, :notnothing, val))
      end
      repl[α] = val
    end
  end
  return IRTools.prewalk!(x -> get(repl, x, x), ir)
end

function stacks!(adj, T)
  forw, stks = forward_stacks!(adj, T)
  IRTools.domorder!(forw)
  back = reverse_stacks!(adj, stks)
  permute!(back, length(back.blocks):-1:1)
  IRTools.domorder!(back)
  return forw, back
end

varargs(m::Method, n) = m.isva ? n - m.nargs + 1 : nothing

function _generate_pullback_via_decomposition(T)
  (m = meta(T)) === nothing && return
  va = varargs(m.method, length(T.parameters))
  forw, back = stacks!(Adjoint(IR(m), varargs = va, normalise = false), T)
  m, forw, back
end

function stacklines(T::Type)
  adj = Adjoint(IR(meta(T)), normalise = false)
  recs = []
  for b in blocks(adj.adjoint), α in alphauses(b)
    push!(recs, IRTools.exprline(adj.primal, Variable(α)))
  end
  return recs
end
