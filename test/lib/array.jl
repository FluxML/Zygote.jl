using LinearAlgebra

# issue 897
@test gradient(x -> sum(sin, Diagonal(x)), ones(2)) == ([0.5403023058681398, 0.5403023058681398],)
