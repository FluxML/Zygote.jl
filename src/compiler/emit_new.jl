#=
TODO:

- All transforms of the form `f(args...)` to `Zygote._forward(ctx, f, args...)`
should simply be removed; Cassette will do the equivalent for you.

- All emitted calls that Zygote does not want to be overdubbed should be wrapped
in `Expr(:nooverdub)` (see Cassette docs for details).
=#

function zygote_pass(::Type{C}, r::Reflection) where C
  T = r.signature
  va = varargs(r.method, length(T.parameters))
  forw, back = stacks!(Adjoint(IRCode(r), varargs = va), T)
  argnames!(r, Symbol("#self#"), :ctx, :f, :args)
  forw = varargs!(r, forw, 3)
  forw = slots!(pis!(inlineable!(forw)))
  return IRTools.update!(r, forw)
end

#=
Example invocation of the above:

julia> using Zygote, Cassette

julia> ctx = Cassette.disablehooks(Zygote.ZygoteContext());

julia> r = Cassette.reflect((typeof(hypot), Int, Int, Int));

julia> Zygote.zygote_pass(typeof(ctx), r)
CodeInfo(
570 1 ─       $(Expr(:meta, :inline))                                          │
    │   %2  = (Zygote._forward)(ctx, Base.Generator, Base.Math.abs2, args)     │
    │   %3  = (Base.getindex)(%2, 1)                                           │
    │         (Base.getindex)(%2, 2)                                           │
    │   %5  = (Zygote._forward)(ctx, Base.Math.sum, %3)                        │
    │   %6  = (Base.getindex)(%5, 1)                                           │
    │         (Base.getindex)(%5, 2)                                           │
    │   %8  = (Zygote._forward)(ctx, Base.Math.sqrt, %6)                       │
    │   %9  = (Base.getindex)(%8, 1)                                           │
    │         (Base.getindex)(%8, 2)                                           │
    │   %11 = (Base.tuple)()                                                   │
    │   %12 = (Zygote.Pullback{Tuple{typeof(hypot),Int64,Int64,Int64},T} where T)(%11)
    │   %13 = (Base.tuple)(%9, %12)                                            │
    └──       return %13                                                       │
)
=#
