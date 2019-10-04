using .Tracker: TrackedArray, TrackedReal

unwrap(x::Union{TrackedArray,TrackedReal}) = Tracker.data(x)

pullback(f, ps::Tracker.Params) = pullback(f, Params(ps))
Tracker.forward(f, ps::Params) = Tracker.forward(f, Tracker.Params(ps))
Tracker.gradient_(f, ps::Params) = Tracker.gradient_(f, Tracker.Params(ps))
