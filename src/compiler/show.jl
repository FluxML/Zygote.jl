funcname(::Type{Type{T}}) where T = string(T)

function funcname(T)
  if isdefined(T, :instance)
    string(T.instance)
  else
    "λ"
  end
end

Base.show(io::IO, j::Pullback{S}) where S = print(io, "∂($(funcname(S.parameters[1])))")
function Base.show(io::IO, P::Type{<:Pullback{S}}) where S
  @isdefined(S) ? print(io, "Pullback{", S, ", ...}") : print(io, "Pullback{S, T}")
end

