module Profile

using Requires
using ..Zygote: Pullback, IdSet, meta, stacklines

function loc(f)
  # TODO perhaps find most general method
  m = first(methods(f))
  :λ, string(m.file), m.line
end

function loc(f::Pullback{T}) where T
  m = meta(T).method
  m.name, string(m.file), m.line
end

# Hack to get inner line numbers for pullbacks wrapped by the `@grad` macro.
function loc_wrapped(x)
  n, f, l = loc(x)
  endswith(f, "src/lib/grad.jl") && length(fields(x)) == 1 ?
    loc(fields(x)[1]) : (n, f, l)
end

function mem(x::Array, seen)
  x in seen && return 0
  push!(seen, x)
  return sizeof(x)
end

fields(x) = map(f -> getfield(x, f), fieldnames(typeof(x)))

function mem(x, seen)
  isbits(x) && return sizeof(x)
  x in seen && return 0
  push!(seen, x)
  sum(x -> mem(x, seen), fields(x))
end

mem(x) = mem(x, IdSet())

struct Node
  func::Symbol
  file::String
  line::Int
  size::Int
  children::Vector{Node}
end

sumsize(cs) = isempty(cs) ? 0 : sum(x->x.size,cs)

Node(func, file, line, cs) = Node(func, file, line, sumsize(cs), cs)

merge(a::Node, b::Node) = Node(a.func, a.file, a.line, a.size+b.size, merge(vcat(a.children, b.children)))

function merge(cs)
  ds = []
  for c in cs
    c.size == 0 && continue
    i = findfirst(x -> (x.func, x.file, x.line) == (c.func, c.file, c.line), ds)
    i === nothing ? push!(ds, c) : (ds[i] = merge(ds[i], c))
  end
  return ds
end

function profile(x, seen)
  [Node(loc_wrapped(x)..., mem(x, seen), [])]
end

function profile(x::Pullback{T}, seen) where T
  ls = []
  for (c, l) in zip(x.t, stacklines(T))
    c isa Union{Integer,Vector{<:Integer}} && continue
    cs = c isa Vector ? merge(vcat(map(x -> profile(x, seen),c)...)) : profile(c, seen)
    push!(ls, Node(loc(x)[1],String(l.file),l.line,cs))
  end
  return merge(ls)
end

function profile(x)
  cs = profile(x, IdSet())
  Node(Symbol(""), "", -1, cs)
end

@init @require Atom="c52e3926-4ff0-5f6e-af25-54175e0327b1" begin
  function tojson(n::Node)
    name, path = Atom.expandpath(string(n.file))
    Dict(:path => path,
         :location => name,
         :func => string(n.func),
         :line => n.line,
         :count => n.size,
         :children => map(tojson, n.children))
  end
  juno(n::Node) = Atom.msg("profile", tojson(n))
end

end
