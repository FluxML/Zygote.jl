using .Tracker: TrackedArray, TrackedReal

unwrap(x::Union{TrackedArray,TrackedReal}) = Tracker.data(x)

pullback(f, ps::Tracker.Params) = pullback(f, Params(ps))
