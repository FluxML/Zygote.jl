module Profile

using Requires
using ..Zygote: Pullback, IdSet, meta, adjoint

function loc(f)
  # TODO perhaps find most general method
  m = first(methods(f))
  string(m.file), m.line
end

function loc(f::Pullback{T}) where T
  m = meta(T).method
  string(m.file), m.line
end

# Hack to get inner line numbers for pullbacks wrapped by the `@grad` macro.
function loc_wrapped(x)
  f, l = loc(x)
  endswith(f, "src/lib/grad.jl") && length(fields(x)) == 1 ?
    loc(fields(x)[1]) : (f, l)
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
  file::String
  line::Int
  size::Int
  children::Vector{Node}
end

function profile(x, seen)
  Node(loc_wrapped(x)..., mem(x, seen), [])
end

function children(x::Pullback)
  cs = []
  for f in x.t
    f isa Vector{<:Integer} && continue # TODO control flow recordings
    f isa AbstractVector ? append!(cs, f) : push!(cs, f)
  end
  return cs
end

function profile(x::Pullback, seen)
  ns = map(x -> profile(x, seen), children(x))
  Node(loc(x)..., sum(x -> x.size, ns), ns)
end

profile(x) = profile(x, IdSet())

@init @require Atom="c52e3926-4ff0-5f6e-af25-54175e0327b1" begin
  function tojson(n::Node)
    name, path = Atom.expandpath(string(n.file))
    Dict(:path => path,
         :location => name,
         :func => "f",
         :line => n.line,
         :count => n.size,
         :children => map(tojson, n.children))
  end
  juno(n::Node) = Atom.msg("profile", tojson(n))
end

end
