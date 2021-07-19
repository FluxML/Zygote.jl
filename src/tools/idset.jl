struct IdSet{T} <: AbstractSet{T}
  dict::IdDict{T,Nothing}
  IdSet{T}() where T = new(IdDict{T,Nothing}())
end

IdSet(xs) = IdSet{eltype(xs)}(xs)

IdSet() = IdSet{Any}()

IdSet{T}(xs) where T = isempty(xs) ? IdSet{T}() : push!(IdSet{T}(), xs...)

Base.push!(s::IdSet{T}, x::T) where T = (s.dict[x] = nothing; s)
Base.delete!(s::IdSet{T}, x::T) where T = (delete!(s.dict, x); s)
Base.in(x, s::IdSet) = haskey(s.dict, x)
Base.eltype(::IdSet{T}) where T = T
Base.collect(s::IdSet) = Base.collect(keys(s.dict))
Base.similar(s::IdSet, T::Type) = IdSet{T}()

@forward IdSet.dict Base.length

Base.iterate(s::IdSet, st...) = iterate(keys(s.dict), st...)
