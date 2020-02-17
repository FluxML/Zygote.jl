# Debugging in Time and Space

Because Zygote generates Julia code for the backwards pass, many of Julia's
normal profiling and performance debugging tools work well on it out of the box.

## Performance Profiling

Julia's [sampling profiler](https://docs.julialang.org/en/v1/manual/profile/) is
useful for understanding performance. We recommend [running the profiler in
Juno](http://docs.junolab.org/latest/man/juno_frontend/#Profiler-1), but the
terminal or [ProfileView.jl](https://github.com/timholy/ProfileView.jl) also
work well.

![](https://i.imgur.com/saYm3Uo.png)

The bars indicate time taken in both the forwards and backwards passes at that
line. The canopy chart on the right shows us each function call as a block,
arranged so that when `f` calls `g`, `g` gets a block just below `f`, which is
bigger the longer it took to run. If we dig down the call stack we'll eventually
find the adjoints for things like `matmul`, which we can click on to view.

![](https://i.imgur.com/ypLQZlu.png)

The trace inside the adjoint can be used to distinguish time taken by the forwards and backwards passes.

## Memory Profiling

Reverse-mode AD typically uses memory proportional to the number of operations
in the program, so long-running programs can also suffer memory usage issues.
Zygote includes a space profiler to help debug these issues. Like the time
profiler, it shows a canopy chart, but this time hovering over it displays the
number of bytes stored by each line of the program.

![](https://i.imgur.com/pd2P4W4.png)

Note that this currently only works inside Juno.

## Reflection

Julia's code and type inference reflection tools can also be useful, though
Zygote's use of closures can make the output noisy. To see the code Julia runs
you should use the low-level `_pullback` method and the pullback it returns.
This will directly show either the derived adjoint code or the code for a custom
adjoint, if there is one.

```julia
julia> using Zygote: Context, _pullback

julia> add(a, b) = a+b

julia> @code_typed _pullback(Context(), add, 1, 2)
CodeInfo(
1 ─ %1 = (Base.getfield)(args, 1)::Int64
│   %2 = (Base.getfield)(args, 2)::Int64
│   %3 = (Base.add_int)(%1, %2)::Int64
│   %4 = (Base.tuple)(%3, $(QuoteNode(∂(add))))::PartialTuple(Tuple{Int64,typeof(∂(add))}, Any[Int64, Const(∂(add), false)])
└──      return %4
) => Tuple{Int64,typeof(∂(add))}

julia> y, back = _pullback(Context(), add, 1, 2)
(3, ∂(add))

julia> @code_typed back(1)
CodeInfo(
1 ─ %1 = (Base.mul_int)(Δ, 1)::Int64
│   %2 = (Base.mul_int)(Δ, 1)::Int64
│   %3 = (Zygote.tuple)(nothing, %1, %2)::PartialTuple(Tuple{Nothing,Int64,Int64}, Any[Const(nothing, false), Int64, Int64])
└──      return %3
) => Tuple{Nothing,Int64,Int64}
```
