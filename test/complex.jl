using Zygote, Test

@test gradient(x -> abs(x)*exp(im*angle(x)), 10+20im)[1] ≈ 1
