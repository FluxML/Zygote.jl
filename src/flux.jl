using .Flux.Tracker: TrackedArray, TrackedReal

unwrap(x::Union{TrackedArray,TrackedReal}) = Flux.data(x)
