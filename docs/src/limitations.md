# Limitations

Zygote aims to support differentiating any code you might write in Julia, but it still has a few limitations. Notably, you might encounter errors when trying to differentiate:
- array mutation
- `try`/`catch` statements
- "foreign call" expressions

In this section, we will introduce examples where each of these errors occurs as well as possible work-arounds.

## Array mutation

Array mutation is by far the most commonly encountered Zygote limitation.

Automatic differentiation (AD) systems like Zygote are built on basic principles of calculus where we encounter _pure_ functions. This means that the function, ``y = f(x)``, does not modify ``x`` and only produces the output ``y`` based on ``x``. If we have a chain of functions, such as ``y = h(g(f(x)))``, we can apply the chain rule to differentiate it. AD systems are built to programmatically apply the chain rule to a series of function calls. Unfortunately, typical programs do not behave this way. We might allocate some memory, `x`, then call a function `y = f!(x)` that modifies `x` to produce the output `y`. This mutating behavior is a _side-effect_ of `f!`. Side-effects are difficult for AD systems to handle, because the must track changes to mutated variables and store older versions of the variable. For these reasons, Zygote does not handle array mutation for now.

Let's explore this with a more concrete example. Here we define a simple mutating function, `f!`, which modifies the elements of its input argument, `x`, in place.
```julia
function f!(x)
  x .= 2 .* x

  return x
end
```
Let's see what happens when we differentiate `f!`
```julia
julia> gradient(rand(3)) do x
         sum(f!(x))
       end
ERROR: Mutating arrays is not supported -- called copyto!(Vector{Float64}, ...)
This error occurs when you ask Zygote to differentiate operations that change
the elements of arrays in-place (e.g. setting values with x .= ...)

Possible fixes:
- avoid mutating operations (preferred)
- or read the documentation and solutions for this error
  https://fluxml.ai/Zygote.jl/latest/limitations

Stacktrace:
  ...
```
We got an error message and a long stacktrace. The error informs us that our code performs array mutation by calling `copyto!` (we might not have directly called this function, but it is being invoked somewhere in the call stack). We see that our code includes `x .= ...` which is given as an example of array mutation. Other examples of mutating operations include:
- setting values (`x .= ...`)
- appending/popping values (`push!(x, v)` / `pop!(x)`)
- calling mutating functions (`mul!(C, A, B)`)

!!! warning

    Non-mutating functions may also use mutation under the hood. This can be done for performance reasons or code re-use.

```julia
function g!(x, y)
  x .= 2 .* y

  return x
end
g(y) = g!(similar(y), y)
```
Here `g` is a "non-mutating function," and it indeed does not mutate `y`, its only argument. But it still allocates a new array and calls `g!` on this array which will result in a mutating operation. You may encounter such functions when working with another package.

Specifically for array mutation, we can use [`Zygote.Buffer`](@ref) to re-write our function. For example, let's fix the function `g!` above.
```julia
function g!(x, y)
  x .= 2 .* y

  return x
end

function g(y)
  x = Zygote.Buffer(y) # Buffer supports syntax like similar
  g!(x, y)
  return copy(x) # this step makes the Buffer immutable (w/o actually copying)
end

julia> gradient(rand(3)) do y
         sum(g(y))
       end
([2.0, 2.0, 2.0],)
```

## Try-catch statements

Any expressions involving `try`/`catch` statements is not supported.
```julia
function tryme(x)
  try
    2 * x
  catch e
    throw(e)
  end
end

julia> gradient(rand(3)) do x
         sum(tryme(x))
       end
ERROR: Compiling Tuple{typeof(tryme), Vector{Float64}}: try/catch is not supported.
Refer to the Zygote documentation for fixes.
https://fluxml.ai/Zygote.jl/latest/limitations

Stacktrace:
  ...
```
Here `tryme` uses a `try`/`catch` statement, and Zygote throws an error when trying to differentiate it as expected. `try`/`catch` expressions are used for error handling, but they are less common in Julia compared to some other languages.

## Foreign call expressions

Foreign call expressions refer to expressions that call external libraries such as code written in C or Fortran. You may want to read more about these calls in the [Julia documentation](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/). Scientific computing libraries in Julia may call established C or Fortran libraries under the hood. Since the underlying code for a foreign call expression is not in Julia, it is not possible for Zygote to differentiate this expression.

Below, we define a function that calls a standard C function, `clock`. This function returns the Unix clock as an `Int32`.
```julia
julia> jclock(x) = ccall(:clock, Int32, ()) * 2
jclock (generic function with 1 method)

julia> jclock(2)
30921278

julia> gradient(jclock, rand())
ERROR: Can't differentiate foreigncall expression
You might want to check the Zygote limitations documentation.
https://fluxml.ai/Zygote.jl/latest/limitations

Stacktrace:
  ...
```
`jclock` will multiply the result of our C function by an argument. When we try to differentiate with respect to this argument, we get an `foreigncall` error.

## Solutions

For all of the errors above, the suggested solutions are similar. You have the following possible work arounds available (in order of preference):
1. avoid the error-inducing operation (e.g. do not use mutating functions)
2. define a [custom `ChainRulesCore.rrule`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/example.html)
3. open an [issue on Zygote](https://github.com/FluxML/Zygote.jl/issues)

Avoiding the operation is simple, just don't do it! If you are using a mutating function, try to use a non-mutating variant. If you are using `try`/`catch` statements, try to use more graceful error handling such as returning `nothing` or another sentinel value. Recall that array mutation can also be avoided by using [`Zygote.Buffer`](@ref) as discussed above.

Sometimes, we cannot avoid expressions that Zygote cannot differentiate, but we may be able to manually derive a gradient. In these cases, you can write [a custom `rrule`](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/example.html) using ChainRules.jl. Please refer to the linked ChainRules documentation for how to do this. _This solution is the only solution available for foreign call expressions._ Below, we provide a custom `rrule` for `jclock`.
```julia
jclock(x) = ccall(:clock, Int32, ()) * x

function ChainRulesCore.rrule(::typeof(jclock), x)
  y = jclock(x)
  pb(ȳ) = (ChainRulesCore.NoTangent(), ȳ * y)

  return y, pb
end

julia> gradient(jclock, rand())
(674298.4243400148,)
```

Lastly, if the code causing problems can be fixed, but it is package code instead of your code, then you should open an issue. For functions built into Julia or its standard libraries, you can open an issue with Zygote.jl or ChainRules.jl. For functions in other packages, you can open an issue with the corresponding package issue tracker.
