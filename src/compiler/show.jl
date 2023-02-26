funcname(::Type{Type{T}}) where T = string(T)

function funcname(T)
  if isdefined(T, :instance)
    string(T.instance)
  else
    "λ"
  end
end

Base.show(io::IO, j::Pullback{S}) where S = print(io, "∂($(funcname(S.parameters[1])))")

