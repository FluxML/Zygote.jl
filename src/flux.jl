using .Flux.Tracker: TrackedArray, TrackedReal

if !usetyped
  unwrap(x::Union{TrackedArray,TrackedReal}) = Flux.data(x)
end
