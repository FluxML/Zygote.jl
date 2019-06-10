using .Tracker: TrackedArray, TrackedReal

if !usetyped
  unwrap(x::Union{TrackedArray,TrackedReal}) = Tracker.data(x)
end

forward(f, ps::Tracker.Params) = forward(f, Params(ps))
