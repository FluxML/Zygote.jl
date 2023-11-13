"""
    dropgrad(x) -> x

Drop the gradient of `x`.

    julia> gradient(2, 3) do a, b
         dropgrad(a)*b
       end
    (nothing, 2)
"""
function dropgrad end

@adjoint dropgrad(x) = dropgrad(x), _ -> nothing

Base.@deprecate dropgrad(x) ChainRulesCore.ignore_derivatives(x)


"""
    ignore() do
      ...
    end

Tell Zygote to ignore a block of code. Everything inside the `do` block will run
on the forward pass as normal, but Zygote won't try to differentiate it at all.
This can be useful for e.g. code that does logging of the forward pass.

Obviously, you run the risk of incorrect gradients if you use this incorrectly.
"""
function ignore end

@adjoint ignore(f) = ignore(f), _ -> nothing

Base.@deprecate ignore(f) ChainRulesCore.ignore_derivatives(f)

"""
    @ignore (...)

Tell Zygote to ignore an expression. Equivalent to `ignore() do (...) end`.
Example:

```julia-repl
julia> f(x) = (y = Zygote.@ignore x; x * y);
julia> f'(1)
1
```
"""
macro ignore(ex)
    return :(Zygote.ignore() do
        $(esc(ex))
    end)
end

using MacroTools: @q

macro nograd(ex)
  Base.depwarn(
    "`Zygote.@nograd myfunc` is deprecated, use `ChainRulesCore.@non_differentiable myfunc(::Any...)` instead.",
    :nograd
  )
  isexpr(ex, :tuple) || (ex = Expr(:tuple, ex))
  blk = @q begin end
  for f in ex.args
    back = MacroTools.@q _ -> ($__source__; nothing)
    push!(blk.args, :(@inline Zygote._pullback(::Context, ::Core.Typeof($(esc(f))), args...) = $(esc(f))(args...), $back))
  end
  return blk
end

# Internal function used by some downstream packages.
# Removing this completely would require some tricky registry changes,
# but leaving it as a vestigial function is much easier.
# See https://github.com/FluxML/Zygote.jl/pull/1328 for more context.
function ∇getindex end
