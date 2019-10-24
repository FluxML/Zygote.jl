# Glossary

Differentiation is a minefield of conflicting and overlapping terminology, partly because the ideas have been re-discovered in many different fields (e.g. calculus and differential geometry, the traditional AD community, deep learning, finance, etc.) Many of these terms are not well-defined and others may disagree on the details. Nevertheless, we aim to at least say how *we* use these terms, which will be helpful when reading over Zygote issues, discussions and source code.

The list is certainly not complete; if you see new terms you'd like defined, or would like to add one yourself, please do open an issue or PR.

**Adjoint**: See *pullback*. Used when defining new pullbacks (i.e. the `@adjoint` macro) since this involves defining the adjoint of the Jacobian, in most cases.

**Backpropagation**: Essentially equivalent to "reverse-mode AD". Used particularly in the machine learning world to refer to simple chains of functions `f(g(h(x)))`, but has generalised beyond that.

**Derivative**: Given a scalar function ``y = f(x)``, the derivative is ``\frac{\partial y}{\partial x}``. "Partial" is taken for granted in AD; there's no interesting distinction between partial and total derivatives for our purposes. It's all in the eye of the beholder.

**Differential**: Given a function ``f(x)``, the linearisation ``\partial f`` such that ``f(x + \epsilon) \approx f(x) + \partial f \epsilon``. This is a generalisation of the derivative since it applies to, for example, vector-to-vector functions (``\partial f`` is a Jacobian) and holomorphic complex functions (``\partial f`` is the first Wirtinger derivative). This is *not*, in general, what Zygote calculates, though differentials can usually be derived from gradients.

**IR**: Intermediate Representation. Essentially source code, but usually lower level – e.g. control flow constructs like loops and branches have all been replaced by `goto`s. The idea is that it's harder for humans to read/write but easier to manipulate programmatically. Worth looking at SSA form as a paradigmatic example.

**Gradient**: See *sensitivity*. There is no technical difference in Zygote's view, though "gradient" sometimes distinguishes the sensitivity we actually want from e.g. the internal ones that Zygote produces as it backpropagates.

**Graph**: ML people tend to think of models as "computation graphs", but this is no more true than any program is a graph. In fact, pretty much anything is a graph if you squint hard enough. This also refers to the data structure that e.g. TensorFlow and PyTorch build to represent your model, but see *trace* for that.

**Pullback**: Given ``y = f(x)`` the function ``\bar x = back(̄\bar y)``. In other words, the function `back` in `y, back = Zygote.pullback(f, x)`.

**Sensitivity**: Used to refer to the gradient ``\bar x = \frac{\partial l}{\partial x}`` with some scalar loss ``l``. In other words, you have a value ``x`` (which need not be scalar) at some point in your program, and ``\bar x`` tells you how you should change that value to decrease the loss. In the AD world, sometimes used to refer to adjoint rules.

**Source to Source Differentiation**: Or Source Code Transformation (SCT). As opposed to *tracing* programs to simplify them, an alternative is to operate directly on a language's source code or IR, generating new source code for pullbacks. This describes Zygote, Swift for TensorFlow, Tapenade and a few other old ADs that worked on C source files. Zygote and Swift are unusual in that they work on in-memory IR rather than text source.

To an extent, tracing ADs can be viewed as source transform of a Wengert list / trace. The key difference is that the trace is a lossy representation of the original semantics, which causes problems with e.g. control flow. Systems which can preserve some of those semantics (e.g. autograph) begin to blur the line here, though they are still not nearly as expressive as language IRs.

**Symbolic Differentiation**: Used to refer to differentiation of "mathematical expressions", that is, things like `3x^2 + sin(x)`. Often distinguished from AD, though this is somewhat arbitrary; you can happily produce a symbolic adjoint for a Wengert list, the only difference being that you're allowed to make variable bindings. So it's really just a special case of AD on an unusually limited language.

**Tape**: This term can refer to pretty much any part of an AD implementation. In particular confusion is caused by conflating the *trace* with the set of values sometimes closed over by a *pullback*. Autograd has a combined trace/closure data structure which is usually described as the tape. On the other hand, PyTorch described their implementation as tape-free because the trace/closure is stored as a DAG rather than a vector, so basically all bets are off here.

**Trace**: A recording of each mathematical operation used by a program, made at runtime and usually forming a Wengert list. Traces may or may not also record actual runtime values (e.g. PyTorch vs. TensorFlow). They can often be treated as an IR and compiled, but are distinguished from true IRs in that they unroll and inline all control flow, functions and data structures. The tracing process can be thought of as a kind of partial evaluation, though tracers are typically much less worried about losing information.

**Vector-Jacobian product**: see *pullback*. So called because all pullbacks are linear functions that can be represented by (left) multiplication with the Jacobian matrix.

**Wengert List**: A set of simple variable assignments and mathematical expressions, forming a directed graph. Can be thought of as a limited programming language with variable bindings and numerical functions but no control flow or data structures. If you *trace* a program for AD it will typically take this form.
