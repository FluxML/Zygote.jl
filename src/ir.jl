Base.show(io::IO, x::SSAValue) = print(io, "%", x.id)

struct Argument
  n::Int
end

Base.show(io::IO, x::Argument) = print(io, "%%", x.n)

struct Phi
  edges::Vector{Int}
  values::Vector{Any}
end

function Base.show(io::IO, phi::Phi)
  print(io, "ϕ(")
  join(io, ["$edge => $value" for (edge,value) in zip(phi.edges,phi.values)], ", ")
  print(io, ")")
end

struct BasicBlock
  succs::Vector{Int}
  preds::Vector{Int}
  first::Int
  last ::Int
end

struct ControlFlowGraph
  blocks::Vector{BasicBlock}
  index::Vector{Int}
end

const CFG = ControlFlowGraph

struct IRCode
  stmts::Vector{Any}
  cfg::ControlFlowGraph
end

function Base.show(io::IO, code::IRCode)
  println(io, "IRCode")
  for (i, x) in enumerate(code.stmts)
    blk = findfirst([1, code.cfg.index...], i)
    blk == 0 || println(io, "$blk:")
    print(io, lpad(i, 3), " | ")
    println(io, x)
  end
end

# Load IR from JSON

function jsonblock(blk)
  BasicBlock(blk["succs"], blk["preds"], blk["stmts"]...)
end

function jsoncfg(cfg)
  CFG(jsonblock.(cfg["blocks"]), cfg["index"])
end

function jsonexpr(x::Associative)
  if haskey(x, "head")
    ex = Expr(Symbol(x["head"]), jsonexpr.(x["args"])...)
    ex.head == :ref && return GlobalRef(getfield(Main, Symbol(ex.args[1])), Symbol(ex.args[2]))
    ex
  elseif haskey(x, "id")
    return SSAValue(x["id"])
  elseif haskey(x, "n")
    return Argument(x["n"])
  elseif haskey(x, "val")
    Expr(:return, jsonexpr(x["val"]))
  elseif haskey(x, "cond")
    Expr(:gotoifnot, jsonexpr(x["cond"]), x["dest"])
  elseif haskey(x, "label")
    GotoNode(x["label"])
  elseif haskey(x, "edges")
    Phi(x["edges"], jsonexpr.(x["values"]))
  else
    x
  end
end

jsonexpr(x) = x

function jsonir(ir)
  IRCode(jsonexpr.(ir["smts"]), jsoncfg(ir["cfg"]))
end
