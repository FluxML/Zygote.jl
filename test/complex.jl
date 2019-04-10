using Zygote, Test

# Gradients are defined as the backpropagation of the number 1. If C is seen as R^2, this means the first column of the transpose of the jacobian.
# For C -> R functions f(x+iy), the Jacobian is (df/dx;df/dy), and so the gradient is df/dx + i df/dy
# For complex valued functions, the gradient is the gradient of the real part. For C -> C functions u+iv=f(x+iy), the gradient is du/dx + i du/dy
# For holomorphic functions, ∇f (as defined above) = du/dx + i du/dy = du/dx - i dv/dx = conj(f'), with f' the complex derivative

fs_C_to_R = (real,
             imag,
             abs,
             abs2,
             z -> abs(z)*cos(im*angle(z)),
             # z->z.re,
             # z->z.im,
             # z->2z.re + 3z.im,
             # z->abs2(z.re+z.im),
             z->abs(cos(exp(z))),
             z->3*real(z)^3-2*imag(z)^5
             )
@testset "C->R" begin
    for f in fs_C_to_R
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote = gradient(f, z)[1]
            ε = 1e-8
            grad_fd = (f(z+ε)-f(z))/ε + im*(f(z+ε*im)-f(z))/ε
            @test abs(grad_zygote - grad_fd) < sqrt(ε)
        end
    end
end

fs_C_to_C_holomorphic = (cos,
                         exp,
                         log,
                         z->z^2,
                         z->(real(z)+im*imag(z))^2,
                         z->real(z)^2 - imag(z)^2 +2im*(real(z)*imag(z)),
                         # z->z.re + z.im*im,
                         z->exp(cos(log(z))),
                         z->abs(z)*exp(im*angle(z)),
                         )
@testset "C->C holomorphic" begin
    for f in fs_C_to_C_holomorphic
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote = gradient(f, z)[1]
            ε = 1e-8
            grad_fd_r = (f(z+ε)-f(z))/ε
            grad_fd_i = (f(z+ε*im)-f(z))/(ε*im)
            @assert abs(grad_fd_r - grad_fd_i) < sqrt(ε) # check the function is indeed holomorphic
            @test abs(grad_zygote - conj(grad_fd_r)) < sqrt(ε)
        end
    end
end


fs_C_to_C_non_holomorphic = (conj,
                             z->abs(z)+0im,
                             z->im*abs(z),
                             z->abs2(z)+0im,
                             z->im*abs2(z),
                             z->z'z,
                             z->conj(z)*z^2,
                             )
@testset "C->C non-holomorphic" begin
    for f in (fs_C_to_C_holomorphic...,fs_C_to_C_holomorphic...)
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote = gradient(f, z)[1]
            ε = 1e-8
            grad_fd = real(f(z+ε)-f(z))/ε + im*real(f(z+ε*im)-f(z))/ε
            @test abs(grad_zygote - grad_fd) < sqrt(ε)
        end
    end
end
