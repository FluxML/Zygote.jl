# Complex Derivatives

Complex numbers add some difficulty to the idea of a "gradient". To talk about `gradient(f, x)` here we need to talk a bit more about `f`.

If `f` returns a real number, things are fairly straightforward. For ``c = x + yi`` and  ``z = f(c)``, we can define the adjoint ``\bar c = \frac{\partial z}{\partial x} + \frac{\partial z}{\partial y}i = \bar x + \bar y i`` (note that ``\bar c`` means gradient, not conjugate, of ``c`` here). It's exactly as if the complex number were just a pair of reals `(re, im)`. This works out of the box.

```julia
julia> gradient(c -> abs2(c), 1+2im)
(2 + 4im,)
```

However, while this is a very pragmatic definition that works great for gradient descent, it's not quite aligned with the mathematical notion of the derivative: i.e. ``f(c + \epsilon) \approx f(c) + \bar c \epsilon``. In general, such a ``\bar c`` is not possible for complex numbers except when `f` is *holomorphic* (or *analytic*). Roughly speaking this means that the function is defined over `c` as if it were a normal real number, without exploiting its complex structure – it can't use `real`, `imag`, `conj`, or anything that depends on these like `abs2` (`abs2(x) = x*conj(x)`). (This constraint also means there's no overlap with the Real case above; holomorphic functions always return complex numbers for complex input.) But most "normal" numerical functions – `exp`, `log`, anything that can be represented by a Taylor series – are fine.

Fortunately it's also possible to get these derivatives; they are the conjugate of the gradients for the real part.

```julia
julia> gradient(x -> real(log(x)), 1+2im)[1] |> conj
0.2 - 0.4im
```

We can check that this function is holomorphic – and thus that the gradient we got out is sensible – by checking the Cauchy-Riemann equations. In other words this should give the same answer:

```julia
julia> -im*gradient(x -> imag(log(x)), 1+2im)[1] |> conj
0.2 - 0.4im
```

Notice that this fails in a non-holomorphic case, `f(x) = log(conj(x))`:

```julia
julia> gradient(x -> real(log(conj(x))), 1+2im)[1] |> conj
0.2 - 0.4im

julia> -im*gradient(x -> imag(log(conj(x))), 1+2im)[1] |> conj
-0.2 + 0.4im
```

In cases like these, all bets are off. The gradient can only be described by a more complex object; either a 2x2 Jacobian (a generalisation of the Real case, where the second column is now non-zero), or by the two Wirtinger derivatives (a generalisation of the holomorphic case, where ``\frac{∂ f}{∂ \bar z}`` is now non-zero). To get these efficiently, as we would a Jacobian, we can just call the backpropagators twice.

```julia
julia> y, back = Zygote.forward(x -> log(conj(x)), 1+2im);

julia> back(1)
(0.2 + 0.4im,)

julia> back(im)
(0.4 - 0.2im,)
```
