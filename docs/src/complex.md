# Complex Differentiation

Complex numbers add some difficulty to the idea of a "gradient". To talk about `gradient(f, x)` here we need to talk a bit more about `f`.

*A note on notation*: We are using ``\bar c`` to mean the gradient of ``c`` here, like we did before. For the complex conjugate of ``c``, we therefore use the notation ``c^*`` and not the ``c'`` Julia code uses, since that could be confused to mean derivative instead, and we also want to distinguish between the complex conjugate and the conjugate transpose. Note however, that whenever we talk about code snippets, `c'` of course still means conjugate (transpose).

If `f` returns a real number, things are fairly straightforward. For ``c = x + yi`` and  ``z = f(c)``, we can define the adjoint ``\bar c = \frac{\partial z}{\partial x} + \frac{\partial z}{\partial y}i = \bar x + \bar y i``. It's exactly as if the complex number were just a pair of reals `(re, im)`. This works out of the box.

```julia
julia> gradient(c -> abs2(c), 1+2im)
(2 + 4im,)
```

However, while this is a very pragmatic definition that works great for gradient descent, it's not quite aligned with the mathematical notion of the derivative: i.e. ``f(c + \epsilon) \approx f(c) + \bar c \epsilon``. In general, such a ``\bar c`` is not possible for complex numbers except when `f` is *holomorphic* (or *analytic*). Roughly speaking this means that the function is defined over `c` as if it were a normal real number, without exploiting its complex structure – it can't use `real`, `imag`, `conj`, or anything that depends on these like `abs2` (`abs2(x) = x*x'`). (This constraint also means there's no overlap with the Real case above; holomorphic functions always return complex numbers for complex input.) But most "normal" numerical functions – `exp`, `log`, anything that can be represented by a Taylor series – are fine.

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

Notice that this fails in a non-holomorphic case, `f(x) = log(x')`:

```julia
julia> gradient(x -> real(log(x')), 1+2im)[1] |> conj
0.2 - 0.4im

julia> -im*gradient(x -> imag(log(x')), 1+2im)[1] |> conj
-0.2 + 0.4im
```

In cases like these, all bets are off. The gradient can only be described with more information; either a 2x2 Jacobian (a generalisation of the Real case, where the second column is now non-zero), or by the two Wirtinger derivatives (a generalisation of the holomorphic case, where ``\frac{\partial f}{\partial z^*}`` is now non-zero). To get these efficiently, as we would a Jacobian, we can just call the backpropagators twice.

```julia
function jacobi(f, x)
  y, back = Zygote.forward(f, x)
  back(1)[1], back(im)[1]
end

function wirtinger(f, x)
  du, dv = jacobi(f, x)
  (du' + im*dv')/2, (du + im*dv)/2
end
```

```julia
julia> wirtinger(x -> 3x^2 + 2x + 1, 1+2im)
(8.0 + 12.0im, 0.0 + 0.0im)

julia> wirtinger(x -> abs2(x), 1+2im)
(1.0 - 2.0im, 1.0 + 2.0im)
```

The gradient definition Zygote uses can also be expressed in terms of the [Wirtinger calculus](https://en.wikipedia.org/wiki/Wirtinger_derivatives) using the operators ``\frac{\partial}{\partial z}`` and ``\frac{\partial}{\partial z^*}``. Since ``f(z)`` is always real, we can use that ``f = \mathrm{Re} f`` as a trick to rewrite the gradient of ``f`` in terms of the Wirtinger derivatives.

```math
f: \mathbb{C} \rightarrow \mathbb{R}, \qquad
f(z) \equiv f(x + iy), \qquad
z \in \mathbb{C}, \ x, y \in \mathbb{R} \\[1.2em]

\bar f \equiv \frac{\partial f}{\partial x} + i \frac{\partial f}{\partial y}
    = 2 \, \frac{\partial(\mathrm{Re}(f))}{\partial z^*}
    = \frac{\partial(f + f^*)}{\partial z^*}
    = \frac{\partial f}{\partial z^*} + \left(\frac{\partial f}{\partial z}\right)^{\!*}
```

Further, we want to study, how these gradients chain together, since the usual chain rule doesn't apply here. We are going to use the relationship we found above, together with the chain rule for Wirtinger derivatives.
Therefore, for the composition of two functions ``f`` and ``w``, one gets the following pullback map, if the inner function ``w`` is holomorphic:

```math
f: \mathbb{C} \rightarrow \mathbb{R}, \qquad
w: \mathbb{C} \rightarrow \mathbb{C} \\[1.2em]

\begin{align*}
\overline{f \circ w}
    &= \frac{\partial (f \circ w)}{\partial z^*} + \left(\frac{\partial (f \circ w)}{\partial z}\right)^{\!*}
    = \frac{\partial f}{\partial w} \frac{\partial w}{\partial z^*}
        + \frac{\partial f}{\partial w^*} \frac{\partial w^*}{\partial z^*}
        + \left( \frac{\partial f}{\partial w} \frac{\partial w}{\partial z}
        + \frac{\partial f}{\partial w^*} \frac{\partial w^*}{\partial z} \right)^{\!*} \\
    &= \left[ \frac{\partial f}{\partial w^*} + \left( \frac{\partial f}{\partial w} \right)^{\!*} \right]
        \left( \frac{\partial w}{\partial z} \right)^{\!*}
    = \overline{f} \cdot \left( \frac{\partial w}{\partial z} \right)^{\!*}
    \qquad \text{if $w(z)$ holomorphic} \Leftrightarrow \frac{\partial w}{\partial z^*} = 0
\end{align*}
```

This nicely explains, why the complex conjugate appears in Zygote's pullback definitions, as pointed out in the [Pullbacks section](../adjoints/#Pullbacks).
If `w` is not holomorphic, the pullback map ``\overline{f} \mapsto \overline{f \circ w}`` is not ``\mathbb{C}``-linear and can therefore not be expressed simply as a multiple of ``\overline{f}``, like in the holomorphic case.

Attention has to be paid, when comparing Zygote to other AD-tools, since they might use different definitions for complex gradients.
