@grad real(x::Complex) = real(x), r̄ -> (r̄ + zero(r̄)*im,)
@grad imag(x::Complex) = imag(x), ī -> (zero(ī) + ī*im,)

# The adjoint of the map z -> g*z is given by y -> g' * y.
# Therefore, for holomorphic functions (for which the differential is given by a complex multiplication),
# the gradient map is given by a multiplication with the conjugate of the derivative (in the holomorphic sense)
@grad log(x::Complex) = log(x), ȳ -> (ȳ/conj(x),)
@grad exp(x::Complex) = exp(x), ȳ -> (ȳ*conj(exp(x)),)
@grad sin(x::Complex) = sin(x), ȳ -> (ȳ*conj(cos(x)),)
@grad cos(x::Complex) = cos(x), ȳ -> (-ȳ*conj(sin(x)),)
