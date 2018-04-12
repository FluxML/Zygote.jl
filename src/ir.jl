using NotInferenceDontLookHere
import NotInferenceDontLookHere: IRCode, CFG, BasicBlock, Argument, ReturnNode,
  GotoIfNot, PhiNode, StmtRange, IncrementalCompact, insert_node!, insert_node_here!,
  compact!, finish, DomTree, construct_domtree
using InteractiveUtils: typesof

Base.getindex(ir::IRCode, x) = NI.getindex(ir, x)
Base.getindex(u::NI.UseRef) = NI.getindex(u)
Base.getindex(r::StmtRange, i) = (r.first:r.last)[i]

for T in [:(NI.UseRefIterator), :(NI.IncrementalCompact)]
  @eval begin
    Base.start(x::$T) = NI.start(x)
    Base.next(x::$T, st) = NI.next(x, st)
    Base.done(x::$T, st) = NI.done(x, st)
  end
end

PhiNode(x, y) = PhiNode(Any[x...], Any[y...])

CFG(bs) = CFG(bs, map(b -> b.stmts.first, bs[2:end]))

Base.show(io::IO, x::SSAValue) = print(io, "%", x.id)

Base.show(io::IO, x::Argument) = print(io, "%%", x.n)

function code_ir(f, T)
  ci = code_typed(f, T, optimize=false)[1][1]
  ssa = compact!(NI.just_construct_ssa(ci, copy(ci.code), length(T.parameters), [NI.NullLineInfo]))
end

macro code_ir(ex)
  isexpr(ex, :call) || error("@code_ir f(args...)")
  f, args = ex.args[1], ex.args[2:end]
  :(code_ir($(esc(f)), typesof($(esc.(args)...))))
end

# Block wrapper

struct Block
  ir::IRCode
  n::Int
end

BasicBlock(b::Block) = b.ir.cfg.blocks[b.n]

Base.range(b::BasicBlock) = b.stmts.first:b.stmts.last
Base.range(b::Block) = range(BasicBlock(b))

insert_node!(b::Block, pos::Int, @nospecialize(typ), @nospecialize(val)) =
  insert_node!(b.ir, pos + range(b)[1] - 1, typ, val)

blocks(ir::IRCode) = [Block(ir, n) for n = 1:length(ir.cfg.blocks)]

# Dominance frontiers

function domfront(cfg, dt = construct_domtree(cfg))
  fronts = [Set{Int}() for _ in cfg.blocks]
  for b = 1:length(cfg.blocks)
    length(cfg.blocks[b].preds) >= 2 || continue
    for p in cfg.blocks[b].preds
      runner = p
      while runner != dt.idoms[b]
        runner == b && break
        push!(fronts[runner], b)
        runner = dt.idoms[runner]
      end
    end
  end
  return fronts
end
