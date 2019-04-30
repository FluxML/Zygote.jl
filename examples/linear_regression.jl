# Initialize environment in current directory
@info("Ensuring example environment instantiated...")
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

@info("Loading Zygote...")
using Zygote, LinearAlgebra

# This example will showcase how we do a simple linear fit with Zygote, making
# use of complex datastructures, a home-grown stochastic gradient descent
# optimizer, and some good old-fashioned math.  We start with the problem
# statement:  We wish to learn the mapping `f(X) -> Y`, where `X` is a matrix
# of vector observations, `f()` is a linear mapping function and `Y` is a
# vector of scalar observations.

# Because we like complex objects, we will define our linear regression as the
# following object:
mutable struct LinearRegression
    # These values will be implicitly learned
    weights::Matrix
    bias::Float64

    # These values will not be learned
    name::String
end
LinearRegression(nparams, name) = LinearRegression(randn(1, nparams), 0.0, name)

# Our linear prediction looks very familiar; w*X + b
function predict(model::LinearRegression, X)
    return model.weights * X .+ model.bias
end

# Our "loss" that must be minimized is the l2 norm  between our current
# prediction and our ground-truth Y
function loss(model::LinearRegression, X, Y)
    return norm(predict(model, X) .- Y, 2)
end


# Our "ground truth" values (that we will learn, to prove that this works)
weights_gt = [1.0, 2.7, 0.3, 1.2]'
bias_gt = 0.4

# Generate a dataset of many observations
X = randn(length(weights_gt), 10000)
Y = weights_gt * X .+ bias_gt

# Add a little bit of noise to `X` so that we do not have an exact solution,
# but must instead do a least-squares fit:
X .+= 0.001.*randn(size(X))


# Now we begin our "training loop", where we take examples from `X`,
# calculate loss with respect to the corresponding entry in `Y`, find the
# gradient upon our model, update the model, and continue.  Before we jump
# in, let's look at what `Zygote.gradient()` gives us:
@info("Building model...")
model = LinearRegression(size(X, 1), "Example")

# Calculate gradient upon `model` for the first example in our training set
@info("Calculating gradient (the first time can take a while to compile...)")
grads = Zygote.gradient(model) do m
    return loss(m, X[:,1], Y[1])
end

# The `grads` object is a Tuple containing one element per argument to
# `gradient()`, so we take the first one to get the gradient upon `model`:
grads = grads[1]

# Because our LinearRegression object is mutable, the gradient holds a
# reference to it, which we peel via `grads[]`:
grads = grads[]

# We now get a `NamedTuple` so we can now do things like `grads.weight`. Let's
# print it out, just to see what it looks like.  Note that while `weights` and
# `bias` have gradients, `name` just naturally has a  gradient of `nothing`,
# because it was not involved in the calculation of the output loss.
@info grads

# Let's define an update rule that will allow us to modify the weights
# of our model a tad bit according to the gradients
function sgd_update!(model::LinearRegression, grads, η = 0.001)
    model.weights .-= η .* grads.weights
    model.bias -= η * grads.bias
end

# Now let's do that for each example in our training set:
@info("Running train loop for $(size(X,2)) iterations")
for idx in 1:size(X, 2)
    grads = Zygote.gradient(m -> loss(m, X[:, idx], Y[idx]), model)[1][]
    sgd_update!(model, grads)
end

# Now let's look at how well we've approximated the ground truth weights/bias:
@info("Ground truth weights: $(weights_gt)")
@info("Learned weights: $(round.(model.weights; digits=3))")
@info("Ground truth bias: $(bias_gt)")
@info("Learned bias: $(round(model.bias; digits=3))")
