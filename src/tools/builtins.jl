macro __new__(T, args...)
  esc(Expr(:new, T, args...))
end

macro __splatnew__(T, args)
  esc(Expr(:splatnew, T, args))
end

# `__new__(T, args...)` must mirror `Expr(:new, T, args...)`, which — unlike
# `splatnew` — permits *fewer* arguments than `T` has fields (the trailing fields
# are then left undefined, exactly as Julia allows for inner constructors such as
# `T(x) = new(x)`). Splatting a runtime tuple into a `:new` expression needs a
# generated function, so we build the `:new` with one argument per element. (#1517)
@generated function __new__(T, args...)
  Expr(:block, Expr(:meta, :inline),
       Expr(:new, :T, (:(args[$i]) for i in 1:length(args))...))
end
@inline __splatnew__(T, args) = @__splatnew__(T, args)

literal_getindex(x, ::Val{i}) where i = getindex(x, i)
literal_indexed_iterate(x, ::Val{i}) where i = Base.indexed_iterate(x, i)
literal_indexed_iterate(x, ::Val{i}, state) where i = Base.indexed_iterate(x, i, state)
