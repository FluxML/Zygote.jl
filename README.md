# Zygote

[![Build Status](https://travis-ci.org/MikeInnes/Zygote.jl.svg?branch=master)](https://travis-ci.org/MikeInnes/Zygote.jl)

Zygote is a *really fast* tool for automatic derivatives in Julia. It works by directly differentiating Julia's internal source code.

## Internals

Zygote provides the following interface for backpropagation:

```julia
y = f(x...)
y, Jᵀ = ∇(f, x...)
(Δx...) = Jᵀ(Δy)
```

Where the `Δ` terms are sensitivities and `Jᵀ` applies the Jacobian. For example, differentiating a matrix multiplication looks as follows.

```julia
W = rand(5, 10)
x = rand(10)

y, Jᵀ = ∇(*, W, x)

ΔW, Δx = Jᵀ(ones(y))
```

Implementing an analytical derivative is simple.

```julia
∇(::typeof(*), a, b) = (a*b, Δ -> (Δ*b', a'*Δ))
```

More complex functions can be implemented in terms of the functions they call.

```julia
f(x) = sin(cos(x))

function ∇(::typeof(f), x)
  c, Jᵀc = ∇(cos, x)
  s, Jᵀs = ∇(sin, c)
  s, Δ -> (Jᵀc(Jᵀs(Δ)[1])[1],)
end
```

More complex functions, including those with control flow, can also be straightforwardly transformed in this way. If you differentiate a power function, for example, code something like this gets generated:

```julia
function pow(x, n)
  r = 1
  for i = 1:n
    r *= x
  end
  return r
end

function ∇(::typeof(pow), x, n)
  r = 1
  Js = []
  for i = 1:n
    r, Jᵀ = ∇(*, r, x)
    push!(Js, Jᵀ)
  end
  return r, function (Δr)
    Δx = 0
    for i = n:-1:1
      Δr, dx = Js[i](Δr)
      Δx += dx
    end
    return Δx
  end
end
```
