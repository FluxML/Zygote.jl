using .Flux
using .Flux.Tracker: TrackedArray, TrackedReal

if !usetyped
  unwrap(x::Union{TrackedArray,TrackedReal}) = Flux.data(x)
end

forward(f, ps::Flux.Tracker.Params) = forward(f, Params(ps))
