var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Zygote-1",
    "page": "Home",
    "title": "Zygote",
    "category": "section",
    "text": "Welcome! Zygote extends the Julia language to support differentiable programming. With Zygote you can write down any Julia code you feel like – including using existing Julia packages – then get gradients and optimise your program. Deep learning, ML and probabilistic programming are all different kinds of differentiable programming that you can do with Zygote.At least, that\'s the idea. We\'re still in beta so expect some adventures."
},

{
    "location": "#Setup-1",
    "page": "Home",
    "title": "Setup",
    "category": "section",
    "text": "Zygote is still moving quickly and it\'s best to work from the development branches. Run this in a Julia session:using Pkg; pkg\"add Zygote#master\""
},

{
    "location": "#Taking-Gradients-1",
    "page": "Home",
    "title": "Taking Gradients",
    "category": "section",
    "text": "Zygote is easy to understand since, at its core, it has a one-function API (forward), along with a few simple conveniences. Before explaining forward, we\'ll look at the higher-level function gradient.gradient calculates derivatives. For example, the derivative of 3x^2 + 2x + 1 is 6x + 2, so when x = 5, dx = 32.julia> using Zygote\n\njulia> gradient(x -> 3x^2 + 2x + 1, 5)\n(32,)gradient returns a tuple, with a gradient for each argument to the function.julia> gradient((a, b) -> a*b, 2, 3)\n(3, 2)This will work equally well if the arguments are arrays, structs, or any other Julia type, but the function should return a scalar (like a loss or objective l, if you\'re doing optimisation / ML).julia> W = rand(2, 3); x = rand(3);\n\njulia> gradient(W -> sum(W*x), W)[1]\n2×3 Array{Float64,2}:\n 0.0462002  0.817608  0.979036\n 0.0462002  0.817608  0.979036\n\njulia> gradient(x -> 3x^2 + 2x + 1, 1//4)\n(7//2,)Control flow is fully supported, including recursion.julia> function pow(x, n)\n         r = 1\n         for i = 1:n\n           r *= x\n         end\n         return r\n       end\npow (generic function with 1 method)\n\njulia> gradient(x -> pow(x, 3), 5)\n(75,)\n\njulia> pow2(x, n) = n <= 0 ? 1 : x*pow(x, n-1)\npow2 (generic function with 1 method)\n\njulia> gradient(x -> pow2(x, 3), 5)\n(75,)Data structures are also supported, including mutable ones like dictionaries. Arrays are currently immutable, though this may change in future.julia> d = Dict()\nDict{Any,Any} with 0 entries\n\njulia> gradient(5) do x\n         d[:x] = x\n         d[:x] * d[:x]\n       end\n(10,)\n\njulia> d[:x]\n5"
},

{
    "location": "#Structs-and-Types-1",
    "page": "Home",
    "title": "Structs and Types",
    "category": "section",
    "text": "Julia makes it easy to work with custom types, and Zygote makes it easy to differentiate them. For example, given a simple Point type:import Base: +, -\n\nstruct Point\n  x::Float64\n  y::Float64\nend\n\na::Point + b::Point = Point(a.x + b.x, a.y + b.y)\na::Point - b::Point = Point(a.x - b.x, a.y - b.y)\ndist(p::Point) = sqrt(p.x^2 + p.y^2)julia> a = Point(1, 2)\nPoint(1.0, 2.0)\n\njulia> b = Point(3, 4)\nPoint(3.0, 4.0)\n\njulia> dist(a + b)\n7.211102550927978\n\njulia> gradient(a -> dist(a + b), a)[1]\n(x = 0.5547001962252291, y = 0.8320502943378437)Zygote\'s default representation of the \"point adjoint\" is a named tuple with gradients for both fields, but this can of course be customised too."
},

{
    "location": "#Gradients-of-ML-models-1",
    "page": "Home",
    "title": "Gradients of ML models",
    "category": "section",
    "text": "It\'s easy to work with even very large and complex models, and there are few ways to do this. Autograd-style models pass around a collection of weights.julia> linear(θ, x) = θ[:W] * x .+ θ[:b]\nlinear (generic function with 1 method)\n\njulia> x = rand(5);\n\njulia> θ = Dict(:W => rand(2, 5), :b => rand(2))\nDict{Any,Any} with 2 entries:\n  :b => [0.0430585, 0.530201]\n  :W => [0.923268 … 0.589691]\n\n# Alternatively, use a named tuple or struct rather than a dict.\n# θ = (W = rand(2, 5), b = rand(2))\n\njulia> θ̄ = gradient(θ -> sum(linear(θ, x)), θ)[1]\nDict{Any,Any} with 2 entries:\n  :b => [1.0, 1.0]\n  :W => [0.628998 … 0.433006]An extension of this is the Flux-style model in which we use call overloading to combine the weight object with the forward pass (equivalent to a closure).julia> struct Linear\n         W\n         b\n       end\n\njulia> (l::Linear)(x) = l.W * x .+ l.b\n\njulia> model = Linear(rand(2, 5), rand(2))\nLinear([0.267663 … 0.334385], [0.0386873, 0.0203294])\n\njulia> dmodel = gradient(model -> sum(model(x)), model)[1]\n(W = [0.652543 … 0.683588], b = [1.0, 1.0])Zygote also support one more way to take gradients, via implicit parameters – this is a lot like autograd-style gradients, except we don\'t have to thread the parameter collection through all our code.julia> W = rand(2, 5); b = rand(2);\n\njulia> linear(x) = W * x .+ b\nlinear (generic function with 2 methods)\n\njulia> grads = gradient(() -> sum(linear(x)), Params([W, b]))\nGrads(...)\n\njulia> grads[W], grads[b]\n([0.652543 … 0.683588], [1.0, 1.0])However, implicit parameters exist mainly for compatibility with Flux\'s current AD; it\'s recommended to use the other approaches unless you need this."
},

{
    "location": "adjoints/#",
    "page": "Custom Adjoints",
    "title": "Custom Adjoints",
    "category": "page",
    "text": ""
},

{
    "location": "adjoints/#Custom-Adjoints-1",
    "page": "Custom Adjoints",
    "title": "Custom Adjoints",
    "category": "section",
    "text": ""
},

{
    "location": "adjoints/#Pullbacks-1",
    "page": "Custom Adjoints",
    "title": "Pullbacks",
    "category": "section",
    "text": "gradient is really just syntactic sugar around the more fundamental function forward.julia> y, back = Zygote.forward(sin, 0.5);\n\njulia> y\n0.479425538604203forward gives two outputs: the result of the original function, sin(0.5), and a pullback, here called back. back implements the gradient computation for sin, accepting a derivative and producing a new one. In mathematical terms, it implements a vector-Jacobian product. Where y = f(x) and the gradient fracpartial lpartial x is written barx, the pullback mathcalB_y computes:barx = fracpartial lpartial x = fracpartial lpartial y fracpartial ypartial x = mathcalB_y(bary)To make this concrete, take the function y = sin(x). fracpartial ypartial x = cos(x), so the pullback is bary cos(x). In other words forward(sin, x) behaves the same asdsin(x) = sin(x), ȳ -> (ȳ * cos(x),)gradient takes a function l = f(x) and assumes l = fracpartial lpartial l = 1 and feeds this in to the pullback. In the case of sin,julia> function gradsin(x)\n         _, back = dsin(x)\n         back(1)\n       end\ngradsin (generic function with 1 method)\n\njulia> gradsin(0.5)\n(0.8775825618903728,)\n\njulia> cos(0.5)\n0.8775825618903728More generallyjulia> function mygradient(f, x...)\n         _, back = Zygote.forward(f, x...)\n         back(1)\n       end\nmygradient (generic function with 1 method)\n\njulia> mygradient(sin, 0.5)\n(0.8775825618903728,)"
},

{
    "location": "adjoints/#Custom-Adjoints-2",
    "page": "Custom Adjoints",
    "title": "Custom Adjoints",
    "category": "section",
    "text": "We can extend Zygote to a new function with the @adjoint function.julia> mul(a, b) = a*b\n\njulia> using Zygote: @adjoint\n\njulia> @adjoint mul(a, b) = mul(a, b), c̄ -> (c̄*b, c̄*a)\n\njulia> gradient(mul, 2, 3)\n(3, 2)It might look strange that we write mul(a, b) twice here. In this case we want to call the normal mul function for the forward pass, but you may also want to modify the forward pass (for example, to capture intermediate results in the pullback)."
},

{
    "location": "adjoints/#Custom-Types-1",
    "page": "Custom Adjoints",
    "title": "Custom Types",
    "category": "section",
    "text": "One good use for custom adjoints is to customise how your own types behave during differentiation. For example, in our Point example we noticed that the adjoint is a named tuple, rather than another point.import Base: +, -\n\nstruct Point\n  x::Float64\n  y::Float64\nend\n\nwidth(p::Point) = p.x\nheight(p::Point) = p.y\n\na::Point + b::Point = Point(width(a) + width(b), height(a) + height(b))\na::Point - b::Point = Point(width(a) - width(b), height(a) - height(b))\ndist(p::Point) = sqrt(width(p)^2 + height(p)^2)julia> gradient(a -> dist(a), Point(1, 2))[1]\n(x = 0.5547001962252291, y = 0.8320502943378437)Fundamentally, this happens because of Zygote\'s default adjoint for getfield.julia> gradient(a -> a.x, Point(1, 2))\n((x = 1, y = nothing),)We can overload this by modifying the getters height and width.julia> @adjoint width(p::Point) = p.x, x̄ -> (Point(x̄, 0),)\n\njulia> @adjoint height(p::Point) = p.y, ȳ -> (Point(0, ȳ),)\n\njulia> Zygote.refresh() # currently needed when defining new adjoints\n\njulia> gradient(a -> height(a), Point(1, 2))\n(Point(0.0, 1.0),)\n\njulia> gradient(a -> dist(a), Point(1, 2))[1]\nPoint(0.4472135954999579, 0.8944271909999159)If you do this you should also overload the Point constructor, so that it can handle a Point gradient (otherwise this function will error).julia> @adjoint Point(a, b) = Point(a, b), p̄ -> (p̄.x, p̄.y)\n\njulia> gradient(x -> dist(Point(x, 1)), 1)\n(0.7071067811865475,)"
},

{
    "location": "adjoints/#Advanced-Adjoints-1",
    "page": "Custom Adjoints",
    "title": "Advanced Adjoints",
    "category": "section",
    "text": "We usually use custom adjoints to add gradients that Zygote can\'t derive itself (for example, because they ccall to BLAS). But there are some more advanced and fun things we can to with @adjoint."
},

{
    "location": "adjoints/#Gradient-Hooks-1",
    "page": "Custom Adjoints",
    "title": "Gradient Hooks",
    "category": "section",
    "text": "julia> hook(f, x) = x\nhook (generic function with 1 method)\n\njulia> @adjoint hook(f, x) = x, x̄ -> (nothing, f(x̄))hook doesn\'t seem that interesting, as it doesn\'t do anything. But the fun part is in the adjoint; it\'s allowing us to apply a function f to the gradient of x.julia> gradient((a, b) -> hook(-, a)*b, 2, 3)\n(-3, 2)We could use this for debugging or modifying gradients (e.g. gradient clipping).julia> gradient((a, b) -> hook(ā -> @show(ā), a)*b, 2, 3)\nā = 3\n(3, 2)Zygote provides both hook and @showgrad so you don\'t have to write these yourself."
},

{
    "location": "adjoints/#Checkpointing-1",
    "page": "Custom Adjoints",
    "title": "Checkpointing",
    "category": "section",
    "text": "A more advanced example is checkpointing, in which we save memory by re-computing the forward pass of a function during the backwards pass. To wit:julia> checkpoint(f, x) = f(x)\ncheckpoint (generic function with 1 method)\n\njulia> @adjoint checkpoint(f, x) = f(x), ȳ -> Zygote._forward(f, x)[2](ȳ)\n\njulia> gradient(x -> checkpoint(sin, x), 1)\n(0.5403023058681398,)If a function has side effects we\'ll see that the forward pass happens twice, as expected.julia> foo(x) = (println(x); sin(x))\nfoo (generic function with 1 method)\n\njulia> gradient(x -> checkpoint(foo, x), 1)\n1\n1\n(0.5403023058681398,)"
},

{
    "location": "adjoints/#Gradient-Reflection-1",
    "page": "Custom Adjoints",
    "title": "Gradient Reflection",
    "category": "section",
    "text": "It\'s easy to check whether the code we\'re running is currently being differentiated.isderiving() = false\n\n@adjoint isderiving() = true, _ -> nothingA more interesting example is to actually detect how many levels of nesting are going on.nestlevel() = 0\n\n@adjoint nestlevel() = nestlevel()+1, _ -> nothingDemo:julia> function f(x)\n         println(nestlevel(), \" levels of nesting\")\n         return x\n       end\nf (generic function with 1 method)\n\njulia> grad(f, x) = gradient(f, x)[1]\ngrad (generic function with 1 method)\n\njulia> f(1);\n0 levels of nesting\n\njulia> grad(f, 1);\n1 levels of nesting\n\njulia> grad(x -> x*grad(f, x), 1);\n2 levels of nesting"
},

{
    "location": "profiling/#",
    "page": "Profiling",
    "title": "Profiling",
    "category": "page",
    "text": ""
},

{
    "location": "profiling/#Profiling-1",
    "page": "Profiling",
    "title": "Profiling",
    "category": "section",
    "text": "WIP"
},

{
    "location": "internals/#",
    "page": "Internals",
    "title": "Internals",
    "category": "page",
    "text": ""
},

{
    "location": "internals/#Internals-1",
    "page": "Internals",
    "title": "Internals",
    "category": "section",
    "text": ""
},

{
    "location": "internals/#What-Zygote-Does-1",
    "page": "Internals",
    "title": "What Zygote Does",
    "category": "section",
    "text": "These notebooks and the Zygote paper provide useful background on Zygote\'s transform; this page is particularly focused on implementation details.Given a function likefunction foo(x)\n  a = bar(x)\n  b = baz(a)\n  return b\nendhow do we differentiate it? The key is that we can differentiate foo if we can differentiate bar and baz. If we assume we can get pullbacks for those functions, the pullback for foo looks as follows.function J(::typeof(foo), x)\n  a, da = J(bar, x)\n  b, db = J(baz, a)\n  return b, function(b̄)\n    ā = db(b̄)\n    x̄ = da(ā)\n    return x̄\n  end\nendThus, where the forward pass calculates x -> a -> b, the backwards takes b̄ -> ā -> x̄ via the pullbacks. The AD transform is recursive; we\'ll differentiate bar and baz in the same way, until we hit a case where gradient is explicitly defined.Here\'s a working example that illustrates the concepts.J(::typeof(sin), x) = sin(x), ȳ -> ȳ*cos(x)\nJ(::typeof(cos), x) = cos(x), ȳ -> -ȳ*sin(x)\n\nfoo(x) = sin(cos(x))\n\nfunction J(::typeof(foo), x)\n  a, da = J(sin, x)\n  b, db = J(cos, a)\n  return b, function(b̄)\n    ā = db(b̄)\n    x̄ = da(ā)\n    return x̄\n  end\nend\n\ngradient(f, x) = J(f, x)[2](1)\n\ngradient(foo, 1)Now, clearly this is a mechanical transformation, so the only remaining thing is to automate it – a small matter of programming."
},

{
    "location": "internals/#Closures-1",
    "page": "Internals",
    "title": "Closures",
    "category": "section",
    "text": "The J function here corresponds to forward in Zygote. However, forward actually a wrapper around the lower level _forward function.julia> y, back = Zygote._forward(sin, 0.5);\n\njulia> back(1)\n(nothing, 0.8775825618903728)Why the extra nothing here? This actually represents the gradient of the function sin. This is often nothing, but when we have closures the function contains data we need gradients for.julia> f = let a = 3; x -> x*a; end\n#19 (generic function with 1 method)\n\njulia> y, back = Zygote._forward(f, 2);\n\njulia> back(1)\n((a = 2,), 3)This is a minor point for the most part, but _forward will come up in future examples."
},

{
    "location": "internals/#Entry-Points-1",
    "page": "Internals",
    "title": "Entry Points",
    "category": "section",
    "text": "We could do this transform with a macro, but don\'t want to require that all differentiable code is annotated. Instead a generated function gets us much of the power of a macro without this annotation, because we can use it to get lowered code for a function. We can then modify the code as we please and return it to implement J(foo, x).julia> foo(x) = baz(bar(x))\nfoo (generic function with 1 method)\n\njulia> @code_lowered foo(1)\nCodeInfo(\n1 ─ %1 = (Main.bar)(x)\n│   %2 = (Main.baz)(%1)\n└──      return %2We convert the code to SSA form using Julia\'s built-in IR data structure, after which it looks like this.julia> Zygote.@code_ir foo(1)\n1 1 ─ %1 = (Main.bar)(_2)::Any\n  │   %2 = (Main.baz)(%1)::Any\n  └──      return %2    (There isn\'t much difference unless there\'s some control flow.)The code is then differentiated by the code in compiler/reverse.jl. You can see the output with @code_adjoint.julia> Zygote.@code_adjoint foo(1)\n1 1 ─ %1  = (Zygote._forward)(_2, Zygote.unwrap, Main.bar)::Any\n  │   %2  = (Base.getindex)(%1, 1)::Any\n  │         (Base.getindex)(%1, 2)::Any\n  │   %4  = (Zygote._forward)(_2, %2, _4)::Any\n  │   %5  = (Base.getindex)(%4, 1)::Any\n  │         (Base.getindex)(%4, 2)::Any\n  │   %7  = (Zygote._forward)(_2, Zygote.unwrap, Main.baz)::Any\n  │   %8  = (Base.getindex)(%7, 1)::Any\n  │         (Base.getindex)(%7, 2)::Any\n  │   %10 = (Zygote._forward)(_2, %8, %5)::Any\n  │   %11 = (Base.getindex)(%10, 1)::Any\n  │         (Base.getindex)(%10, 2)::Any\n  └──       return %11\n  1 ─ %1  = Δ()::Any\n1 │   %2  = (@12)(%1)::Any\n  │   %3  = (Zygote.gradindex)(%2, 1)::Any\n  │   %4  = (Zygote.gradindex)(%2, 2)::Any\n  │         (@9)(%3)::Any\n  │   %6  = (@6)(%4)::Any\n  │   %7  = (Zygote.gradindex)(%6, 1)::Any\n  │   %8  = (Zygote.gradindex)(%6, 2)::Any\n  │         (@3)(%7)::Any\n  │   %10 = (Zygote.tuple)(nothing, %8)::Any\n  └──       return %10\n, [1])This code is quite verbose, mainly due to all the tuple unpacking (gradindex is just like getindex, but handles nothing gracefully). The are two pieces of IR here, one for the modified forward pass and one for the pullback closure. The @ nodes allow the closure to refer to values from the forward pass, and the Δ() represents the incoming gradient ȳ. In essence, this is just what we wrote above by hand for J(::typeof(foo), x).compiler/emit.jl lowers this code into runnable IR (e.g. by turning @ references into getfields and stacks), and it\'s then turned back into lowered code for Julia to run."
},

{
    "location": "internals/#Closure-Conversion-1",
    "page": "Internals",
    "title": "Closure Conversion",
    "category": "section",
    "text": "There are no closures in lowered Julia code, so we can\'t actually emit one directly in lowered code. To work around this we have a trick: we have a generic struct likestruct Pullback{F}\n  data\nendWe can put whatever we want in data, and the F will be the signature for the original call, like Tuple{typeof(foo),Int}. When the pullback gets called it hits another generated function which emits the pullback code.In hand written code this would look like:struct Pullback{F}\n  data\nend\n\nfunction J(::typeof(foo), x)\n  a, da = J(sin, x)\n  b, db = J(cos, a)\n  return b, Pullback{typeof(foo)}((da, db))\nend\n\nfunction(p::Pullback{typeof(foo)})(b̄)\n  da, db = p.data[1], p.data[2]\n  ā = db(b̄)\n  x̄ = da(ā)\n  return x̄\nend"
},

{
    "location": "internals/#Debugging-1",
    "page": "Internals",
    "title": "Debugging",
    "category": "section",
    "text": "Say some of our code is throwing an error.bad(x) = x\n\nZygote.@adjoint bad(x) = x, _ -> error(\"bad\")\n\nfoo(x) = bad(sin(x))\n\ngradient(foo, 1) # error!Zygote can usually give a stacktrace pointing right to the issue here, but in some cases there are compiler crashes that make this harder. In these cases it\'s best to (a) use _forward and (b) take advantage of Zygote\'s recursion to narrow down the problem function.julia> y, back = Zygote._forward(foo, 1);\n\njulia> back(1) # just make up a value here, it just needs to look similar to `y`\nERROR: bad\n\n# Ok, so we try functions that foo calls\n\njulia> y, back = Zygote._forward(sin, 1);\n\njulia> back(1)\n(nothing, 0.5403023058681398)\n\n# Looks like that\'s fine\n\njulia> y, back = Zygote._forward(bad, 1);\n\njulia> back(1) # ok, here\'s our issue. Lather, rinse, repeat.\nERROR: badOf course, our goal is that you never have to do this, but until Zygote is more mature it can be a useful way to narrow down test cases."
},

]}
