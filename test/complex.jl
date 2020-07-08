using Zygote, Test, LinearAlgebra

@test gradient(x -> real(abs(x)*exp(im*angle(x))), 10+20im)[1] ≈ 1
@test gradient(x -> imag(real(x)+0.3im), 0.3)[1] ≈ 0
@test gradient(x -> imag(conj(x)+0.3im), 0.3)[1] ≈ -1im
@test gradient(x -> abs((imag(x)+0.3)), 0.3)[1] == 1im

@test gradient(a -> real((a*conj(a))), 0.3im)[1] == 0.6im
@test gradient(a -> real((a.*conj(a))), 0.3im)[1] == 0.6im
@test gradient(a -> real(([a].*conj([a])))[], 0.3im)[1] == 0.6im
@test gradient(a -> real(([a].*conj.([a])))[], 0.3im)[1] == 0.6im
@test gradient(a -> real.(([a].*conj.([a])))[], 0.3im)[1] == 0.6im

@test gradient(x -> norm((im*x) ./ (im)), 2)[1] == 1
@test gradient(x -> norm((im) ./ (im*x)), 2)[1] == -1/4
@test gradient(x -> real(det(x)), [1 2im; 3im 4])[1] ≈ [4 3im; 2im 1]
@test gradient(x -> real(logdet(x)), [1 2im; 3im 4])[1] ≈ [4 3im; 2im 1]/10
@test gradient(x -> real(logabsdet(x)[1]), [1 2im; 3im 4])[1] ≈ [4 3im; 2im 1]/10

# https://github.com/FluxML/Zygote.jl/issues/705
@test gradient(x -> imag(sum(exp, x)), [1,2,3])[1] ≈ im .* exp.(1:3)
@test gradient(x -> imag(sum(exp, x)), [1+0im,2,3])[1] ≈ im .* exp.(1:3)

fs_C_to_R = (real,
             imag,
             abs,
             abs2,
             z -> abs(z)*cos(im*angle(z)),
             z->abs(cos(exp(z))),
             z->3*real(z)^3-2*imag(z)^5
             )
@testset "C->R" begin
    for f in fs_C_to_R
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote = gradient(real∘f, z)[1]
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
                         z->exp(cos(log(z))),
                         z->abs(z)*exp(im*angle(z)),
                         )
@testset "C->C holomorphic" begin
    for f in fs_C_to_C_holomorphic
        for z in (1.0+2.0im, -2.0+pi*im)
            grad_zygote = gradient(real∘f, z)[1]
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
            grad_zygote = gradient(real∘f, z)[1]
            ε = 1e-8
            grad_fd = real(f(z+ε)-f(z))/ε + im*real(f(z+ε*im)-f(z))/ε
            @test abs(grad_zygote - grad_fd) < sqrt(ε)
        end
    end
end
