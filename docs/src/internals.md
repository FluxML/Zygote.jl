# Internals

## What Zygote Does

[These notebooks](https://github.com/MikeInnes/diff-zoo) and the [Zygote paper](https://arxiv.org/abs/1810.07951) provide useful background on Zygote's transform; this page is particularly focused on implementation details.

Before we think about AD, we'll consider some simple cases. We can start by defining a function that produces pullbacks, `J`, explicitly for some simple functions.

```julia
J(::typeof(sin), x) = sin(x), ȳ -> ȳ*cos(x)
J(::typeof(cos), x) = cos(x), ȳ -> -ȳ*sin(x)
J(::typeof(*), a, b) = a*b, c̄ -> (b*c̄, a*c̄)
```

Now we can call `J` to take a gradient.

```julia
gradient(f, x...) = J(f, x...)[2](1)

gradient(sin, 1) # (0.540,)

gradient(*, 2, 3) # (3, 2)
```

Now consider a composite function that calls two simple ones:

```julia
function foo(x)
  a = sin(x)
  b = cos(a)
  return b
end
```

We can easily differentiate `foo` if we can differentiate the functions it calls. If we can get pullbacks via `J`, the pullback for `foo` looks as follows. Where the forward pass calculates `x -> a -> b`, the backwards takes `b̄ -> ā -> x̄` via the pullbacks.

```julia
function J(::typeof(foo), x)
  a, da = J(sin, x)
  b, db = J(cos, a)
  return b, function(b̄)
    ā, = db(b̄)
    x̄, = da(ā)
    return x̄
  end
end

gradient(foo, 1) # (-0.403,)
```

Things get just a little more complex when control flow is involved. You can see that the derived adjoint for `pow` mirrors the original function, except that the loop runs in reverse. The easiest way to see why it looks like this is to imagine unrolling the loop `n` times, working out the adjoint, and then turning it back into a loop.

```julia
function pow(x, n) # x^n
  r = 1
  for _ = 1:n
    r *= x
  end
  return r
end

function J(::typeof(pow), x, n)
  r = 1
  Js = []
  for i = 1:n
    r, back = J(*, r, x)
    push!(Js, back)
  end
  return r, function(r̄)
    x̄ = 0
    for i = n:-1:1
      r̄, x̄′ = Js[i](r̄)
      x̄ += x̄′
    end
    return (x̄, 0)
  end
end

gradient(pow, 2, 3) # (12, 0)
```

Despite being reasonably fiddly, this is a fully mechanical transformation, so the only remaining thing is to automate it – a small matter of programming.

## Closures

The `J` function here corresponds to `pullback` in Zygote. However, `pullback` is actually a wrapper around the lower level `_pullback` function.

```julia
julia> y, back = Zygote._pullback(sin, 0.5);

julia> back(1)
(nothing, 0.8775825618903728)
```

Why the extra `nothing` here? This actually represents the gradient of the function `sin`. This is often `nothing`, but when we have closures the function contains data we need gradients for.

```julia
julia> f = let a = 3; x -> x*a; end
#19 (generic function with 1 method)

julia> y, back = Zygote._pullback(f, 2);

julia> back(1)
((a = 2,), 3)
```

This is a minor point for the most part, but `_pullback` will come up in future examples.

## Entry Points

You might notice that Zygote is, in effect, _just a macro_. We could happily implement Zygote by writing definitions like

```julia
@differentiable foo(x) = sin(cos(x))
```

which would expand to generate an appropriate overload to `J`. As long as every function we want to differentiate is annotated, this will work just fine. However, it's obviously not ideal to have to annotate every function inside every Julia package in order to make it differentiable.

This is where generated functions come in. Making `J` a [generated function](https://github.com/FluxML/Zygote.jl/blob/daf1032488a2cd1fc739bc95a9fc05f93f90f2b6/src/compiler/interface2.jl#L3) allows us to apply the Zygote macro on an as-needed basis; calling `J(f, x...)` looks up the code for `f(x...)`, transforms it, and then behaves as if you had defined `J` for that specific function ahead of time.

When we look up the code, we actually get *lowered* (desugared) code rather than an AST.

```julia
julia> foo(x) = baz(bar(x))
foo (generic function with 1 method)

julia> @code_lowered foo(1)
CodeInfo(
1 ─ %1 = (Main.bar)(x)
│   %2 = (Main.baz)(%1)
└──      return %2
```

We convert the code to SSA form using Julia's built-in IR data structure, after which it looks like this.

```julia
julia> Zygote.@code_ir foo(1)
1 1 ─ %1 = (Main.bar)(_2)::Any
  │   %2 = (Main.baz)(%1)::Any
  └──      return %2    
```

(There isn't much difference unless there's some control flow.)

The code is then differentiated by the code in `compiler/reverse.jl`. You can see the output with `@code_adjoint`.

```julia
julia> Zygote.@code_adjoint foo(1)
1 1 ─ %1  = (Zygote._pullback)(_2, Zygote.unwrap, Main.bar)::Any
  │   %2  = (Base.getindex)(%1, 1)::Any
  │         (Base.getindex)(%1, 2)::Any
  │   %4  = (Zygote._pullback)(_2, %2, _4)::Any
  │   %5  = (Base.getindex)(%4, 1)::Any
  │         (Base.getindex)(%4, 2)::Any
  │   %7  = (Zygote._pullback)(_2, Zygote.unwrap, Main.baz)::Any
  │   %8  = (Base.getindex)(%7, 1)::Any
  │         (Base.getindex)(%7, 2)::Any
  │   %10 = (Zygote._pullback)(_2, %8, %5)::Any
  │   %11 = (Base.getindex)(%10, 1)::Any
  │         (Base.getindex)(%10, 2)::Any
  └──       return %11
  1 ─ %1  = Δ()::Any
1 │   %2  = (@12)(%1)::Any
  │   %3  = (Zygote.gradindex)(%2, 1)::Any
  │   %4  = (Zygote.gradindex)(%2, 2)::Any
  │         (@9)(%3)::Any
  │   %6  = (@6)(%4)::Any
  │   %7  = (Zygote.gradindex)(%6, 1)::Any
  │   %8  = (Zygote.gradindex)(%6, 2)::Any
  │         (@3)(%7)::Any
  │   %10 = (Zygote.tuple)(nothing, %8)::Any
  └──       return %10
, [1])
```

This code is quite verbose, mainly due to all the tuple unpacking (`gradindex` is just like `getindex`, but handles `nothing` gracefully). There are two pieces of IR here, one for the modified pullback pass and one for the pullback closure. The `@` nodes allow the closure to refer to values from the pullback pass, and the `Δ()` represents the incoming gradient `ȳ`. In essence, this is just what we wrote above by hand for `J(::typeof(foo), x)`.

`compiler/emit.jl` lowers this code into runnable IR (e.g. by turning `@` references into `getfield`s and stacks), and it's then turned back into lowered code for Julia to run.

## Closure Conversion

There are no closures in lowered Julia code, so we can't actually emit one directly in lowered code. To work around this we have a trick: we have a generic struct like

```julia
struct Pullback{F}
  data
end
```

We can put whatever we want in `data`, and the `F` will be the signature for the *original* call, like `Tuple{typeof(foo),Int}`. When the pullback gets called it hits [another generated function](https://github.com/FluxML/Zygote.jl/blob/daf1032488a2cd1fc739bc95a9fc05f93f90f2b6/src/compiler/interface2.jl#L15) which emits the pullback code.

In hand written code this would look like:

```julia
struct Pullback{F}
  data
end

function J(::typeof(foo), x)
  a, da = J(sin, x)
  b, db = J(cos, a)
  return b, Pullback{typeof(foo)}((da, db))
end

function(p::Pullback{typeof(foo)})(b̄)
  da, db = p.data[1], p.data[2]
  ā = db(b̄)
  x̄ = da(ā)
  return x̄
end
```

## Debugging

Say some of our code is throwing an error.

```julia
bad(x) = x

Zygote.@adjoint bad(x) = x, _ -> error("bad")

foo(x) = bad(sin(x))

gradient(foo, 1) # error!
```

Zygote can usually give a stacktrace pointing right to the issue here, but in some cases there are compiler crashes that make this harder. In these cases it's best to (a) use `_pullback` and (b) take advantage of Zygote's recursion to narrow down the problem function.

```julia
julia> y, back = Zygote._pullback(foo, 1);

julia> back(1) # just make up a value here, it just needs to look similar to `y`
ERROR: bad

# Ok, so we try functions that foo calls

julia> y, back = Zygote._pullback(sin, 1);

julia> back(1)
(nothing, 0.5403023058681398)

# Looks like that's fine

julia> y, back = Zygote._pullback(bad, 1);

julia> back(1) # ok, here's our issue. Lather, rinse, repeat.
ERROR: bad
```

Of course, our goal is that you never have to do this, but until Zygote is more mature it can be a useful way to narrow down test cases.
