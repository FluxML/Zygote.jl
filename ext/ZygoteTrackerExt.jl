module ZygoteTrackerExt

using Zygote
using Tracker: Tracker, TrackedArray, TrackedReal

Zygote.unwrap(x::Union{TrackedArray,TrackedReal}) = Tracker.data(x)

Zygote.pullback(f, ps::Tracker.Params) = pullback(f, ZygtParams(ps))
Tracker.forward(f, ps::Params) = Tracker.forward(f, Tracker.Params(ps))
Tracker.gradient_(f, ps::Params) = Tracker.gradient_(f, Tracker.Params(ps))

end
