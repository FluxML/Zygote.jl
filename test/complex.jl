using Zygote, Test

@test gradient(x -> real(abs(x)*exp(im*angle(x))), 10+20im)[1] ≈ 1
@test gradient(x -> imag(real(x)+0.3im), 0.3)[1] ≈ 0
@test gradient(x -> imag(conj(x)+0.3im), 0.3)[1] ≈ -1im
@test gradient(x -> abs((imag(x)+0.3)), 0.3)[1] == 1im
