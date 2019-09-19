# Internals

## What Zygote Does

[These notebooks](https://github.com/MikeInnes/diff-zoo) and the [Zygote paper](https://arxiv.org/abs/1810.07951) provide useful background on Zygote's transform; this page is particularly focused on implementation details.

Given a function like

```julia
function foo(x)
  a = bar(x)
  b = baz(a)
  return b
end
```

how do we differentiate it? The key is that we can differentiate `foo` if we can differentiate `bar` and `baz`. If we assume we can get pullbacks for those functions, the pullback for `foo` looks as follows.

```julia
function J(::typeof(foo), x)
  a, da = J(bar, x)
  b, db = J(baz, a)
  return b, function(b̄)
    ā = db(b̄)
    x̄ = da(ā)
    return x̄
  end
end
```

Thus, where the pullback pass calculates `x -> a -> b`, the backwards takes `b̄ -> ā -> x̄` via the pullbacks. The AD transform is recursive; we'll differentiate `bar` and `baz` in the same way, until we hit a case where gradient is explicitly defined.

Here's a working example that illustrates the concepts.

```julia
J(::typeof(sin), x) = sin(x), ȳ -> ȳ*cos(x)
J(::typeof(cos), x) = cos(x), ȳ -> -ȳ*sin(x)

foo(x) = sin(cos(x))

function J(::typeof(foo), x)
  a, da = J(sin, x)
  b, db = J(cos, a)
  return b, function(b̄)
    ā = db(b̄)
    x̄ = da(ā)
    return x̄
  end
end

gradient(f, x) = J(f, x)[2](1)

gradient(foo, 1)
```

Now, clearly this is a mechanical transformation, so the only remaining thing is to automate it – a small matter of programming.

## Closures

The `J` function here corresponds to `pullback` in Zygote. However, `pullback` actually a wrapper around the lower level `_pullback` function.

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

We could do this transform with a macro, but don't want to require that all differentiable code is annotated. Instead a [generated function](https://github.com/FluxML/Zygote.jl/blob/daf1032488a2cd1fc739bc95a9fc05f93f90f2b6/src/compiler/interface2.jl#L3) gets us much of the power of a macro without this annotation, because we can use it to get lowered code for a function. We can then modify the code as we please and return it to implement `J(foo, x)`.

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

This code is quite verbose, mainly due to all the tuple unpacking (`gradindex` is just like `getindex`, but handles `nothing` gracefully). The are two pieces of IR here, one for the modified pullback pass and one for the pullback closure. The `@` nodes allow the closure to refer to values from the pullback pass, and the `Δ()` represents the incoming gradient `ȳ`. In essence, this is just what we wrote above by hand for `J(::typeof(foo), x)`.

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
