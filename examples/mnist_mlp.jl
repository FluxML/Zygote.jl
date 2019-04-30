# Initialize environment in current directory
@info("Ensuring example environment instantiated...")
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

@info("Loading Zygote and Flux...")
using Zygote, Flux, Random, Statistics
using Flux.Data.MNIST

# We're going to showcase how to use Zygote with Flux; we'll create a simple
# Multi-Layer Perceptron network to do digit classification upon the MNIST
# dataset.  We start with some setup that is ripped straight from the Flux
# model zoo:

# First, we load the MNIST images and flatten them into a giant matrix:
@info("Loading dataset...")
X = hcat(float.(reshape.(MNIST.images(), :))...)

# Load labels as well, one-hot encoding them
Y = float.(Flux.onehotbatch(MNIST.labels(), 0:9))

# Do the same for the test data/labels:
X_test = hcat(float.(reshape.(MNIST.images(:test), :))...)
Y_test = float.(Flux.onehotbatch(MNIST.labels(:test), 0:9))

@info("Constructing MLP model...")
model = Chain(
    Dense(28^2, 32, relu),
    Dense(32, 10),
    softmax,
)

# Until Flux drops Tracker as its default Automatic Differentiation library,
# strip it out with this line:
model = Flux.mapleaves(Flux.data, model)

# Our loss is the classical multiclass crossentropy loss
loss(model, X, Y) = Flux.crossentropy(model(X), Y)

# Helper function to calculate accuracy of our model
accuracy(model, X, Y) = mean(Flux.onecold(model(X)) .== Flux.onecold(Y))


# Recursive zygote update method, this is the general recursion case:
function zyg_update!(opt, model, updates)
	# If this `model` node has no fields, then just return it
    if nfields(model) == 0
        return model
    end

	# If it does have fields, recurse into them:
    for field_idx in 1:nfields(model)
        zyg_update!(opt, getfield(model, field_idx), getfield(updates, field_idx))
    end

    # In the end, return the `model`
    return model
end
# If the `updates` is set to `Nothing`, then just return `model`; this means
# that there were no changes to be applied to this piece of the model.
zyg_update!(opt, model, updates::Nothing) = model

# If `model` is an `AbstractArray` and `updates` is too, then apply our Flux
# optimizer to the incoming gradients and apply them to the model!
function zyg_update!(opt, model::AbstractArray, updates::AbstractArray)
    # Sub off to Flux's ADAM optimizer
    Flux.Optimise.apply!(opt, model, updates)
    return model .-= updates
end


# We will train for a number of epochs, with minibatches, using the `ADAM`
# optimizer to nudge our weights toward perfection.
opt = ADAM(0.001)
num_epochs = 10
@info("Training for $(num_epochs) epochs...")
for epoch_idx in 1:num_epochs
    # "global" here to dodgescoping issues with for loops at top-level
    global X, Y, model

    # Shuffle the data each epoch:
    perm = shuffle(1:size(X,2))
    X = X[:, perm]
    Y = Y[:, perm]

    # Iterate over batches
    batch_size = 512
    batch_idxs = 1:batch_size:(size(X,2) - batch_size)
    for bidx in batch_idxs
        # Calculate gradients upon the model for this batch
        grads = Zygote.gradient(model) do model
            return loss(model, X[:, bidx:bidx+batch_size],
                               Y[:, bidx:bidx+batch_size])
        end

        # Peel outer Tuple to access gradient of first parameter
        grads = grads[1]

        # Apply recursive update to our model:
        zyg_update!(opt, model, grads)
    end

    # After each epoch, report our accuracy on the test set:
    acc = accuracy(model, X_test, Y_test)
    @info("[$(epoch_idx)] Accuracy: $(round(100*acc; digits=1))%")
end
