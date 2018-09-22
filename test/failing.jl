using Zygote, Test, LinearAlgebra
using Zygote: Params, gradient, forward

x = randn(3)
v = randn(3)
H = randn(3,3); H = H+H'

f(x) = (H = [ -0.466054  -1.67757   0.227333
 -1.67757   -0.7407    1.90789
  0.227333   1.90789  -1.10871 ]; transpose(x)*(H*x))      # i'H*i function to take hessian of

val,back = forward(f,x)
@test_skip gradient(f,x)
@test_skip back(x)

g(x)     = 0.5*x'*(H*x)
val,back = forward(g,x)
@test_skip back(x)

@test_skip gradtest(2) do x
  H = [1 0.5; 0.5 2]
  0.5*(x'*(H*x))
end
