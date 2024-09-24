# Design Limitations

Zygote aims to support differentiating any Julia code, but it still has a few limitations.
Notably, you might encounter errors when trying to differentiate:
- array mutation,
- `try`/`catch` statements,
- "foreign call" expressions.

This section gives examples where each of these errors occurs, as well as possible work-arounds.

Below, it also describes some known bugs in expressions Zygote ought to be able to handle.

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

    Non-mutating functions might also use mutation under the hood. This can be done for performance reasons or code re-use.

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

Code containting try-catch blocks can be differentiated as long as no exception is actually thrown.

```julia
julia> function safe_sqrt(x)
           try
               sqrt(x)
           catch
               0.
           end
       end
safe_sqrt (generic function with 1 method)

julia> gradient(safe_sqrt, 4.)
(0.25,)

julia> val, pull = pullback(safe_sqrt, -1.)
(0.0, Zygote.var"#76#77"{Zygote.Pullback{Tuple{typeof(safe_sqrt), Float64}, Any}}(∂(safe_sqrt)))

julia> pull(1.)
ERROR: Can't differentiate function execution in catch block at #= REPL[2]:3 =#.
Stacktrace:
```

Here, the `safe_sqrt` function catches DomainError from the sqrt call when the input is out of domain and safely returns 0. Zygote is able to differentiate the function when no error is thrown by the sqrt call, but fails to differentiate when the control flow goes through the catch block.

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

# Solutions

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


# Known Issues

Zygote's issue tracker has the current list of open [bugs](https://github.com/FluxML/Zygote.jl/issues?q=is%3Aissue+is%3Aopen+label%3Abug). There are some general principles about things you may wish to avoid if you can:

## `mutable struct`s

Zygote has limited support for mutation, and in particular will allow you to change a field in some `mutable struct X; a; b; end` by setting `x.a = val`.

However, this has [many limitations](https://github.com/FluxML/Zygote.jl/issues?q=is%3Aissue+is%3Aopen+mutable+struct) and should be avoided if possible.

The simple solution is to use only immutable `struct`s. 

If you need to modify them, using something like `@set` from [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) should work well. This returns a new object, but does not have side-effects on other copies of it. 

## Re-using variable names

It is common to accumulate values in a loop by re-binding the same variable name to a new value
many times, for example:
```
function mysum(x::Real, n::Int)
  tot = 0.0
  for i in 1:n
    tot += x^n  # binds symbol `tot` to new value
  end
  return tot
end
```
However, sometimes such re-binding confuses Zygote, especially if the type of the value changes. Especially if the variable is "boxed", as will happen if you re-bind from within a closure (such as the function created by a `do` block).

## Second derivatives

In principle Zygote supports taking derivatives of derivatives. There are, however, a few problems:
* Quite a few of its rules are not written in a way that is itself differentiable. For instance they may work by making an array then writing into it, which is mutation of the sort forbidden above. 
* The complexity of the code grows rapidly, as Zygote differentiates its own un-optimised output.
* Reverse mode over reverse mode is seldom the best algorithm.

The issue tracker has a label for [second order](https://github.com/FluxML/Zygote.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22second+order%22), which will outline where the bodies are buried.

Often using a different AD system over Zygote is a better solution.
This is what [`hessian`](@ref) does, using ForwardDiff over Zygote, but other combinations are possible.
(Note that rules defined here mean that Zygote over ForwardDiff is translated to ForwardDiff over ForwardDiff.)
