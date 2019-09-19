using .Tracker: TrackedArray, TrackedReal

if !usetyped
  unwrap(x::Union{TrackedArray,TrackedReal}) = Tracker.data(x)
end

pullback(f, ps::Tracker.Params) = pullback(f, Params(ps))
