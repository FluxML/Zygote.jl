# AGENTS.md

This file provides guidance to AI coding agents (e.g. Claude Code, claude.ai/code) when working with code in this repository.

Zygote is a **source-to-source reverse-mode automatic differentiation (AD)** system for Julia. Instead of tracing or operator overloading, it hooks into the Julia compiler: given a function, it fetches the *lowered* IR, mechanically transforms it into a forward pass plus a "pullback" closure, and emits that as runnable code. It is the AD backend for Flux.

## Commands

The package uses a Julia 1.11+ **workspace** (`[workspace]` in `Project.toml`): `test/Project.toml` shares the top-level manifest, so the test environment is `--project=test`.

```bash
# Develop in the REPL
julia --project=.            # then: using Zygote

# Instantiate the test environment (first time)
julia --project=test -e 'using Pkg; Pkg.instantiate()'

# Run the full test suite (CUDA tests auto-skip unless a GPU is present)
julia --project=test test/runtests.jl

# Run a single test item by name (regex match on @testitem name)
julia --project=test -e 'using ReTestItems, Zygote; runtests(Zygote; name="compiler")'

# Run a single test file
julia --project=test -e 'using ReTestItems; runtests("test/compiler_tests.jl")'

# Build docs / run doctests
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
julia --project=docs -e 'using Documenter, Zygote; doctest(Zygote)'
```

Tests use **ReTestItems**: each `test/*_tests.jl` is one `@testitem "<name>" begin ... end`. Item names (use these with `name=`): `compiler`, `chainrules`, `complex`, `features`, `interface`, `lib`, `structures`, `tools`, `utils`, `forward`, `deprecated`, `gradcheck pt. 1`–`4`, `cuda` (GPU-only), `pythoncall` (skipped). The `gradcheck pt. N` items depend on the shared `@testsetup module GradCheckSetup` in `test/gradcheck_testsetup.jl` (finite-difference gradient checking helpers: `gradcheck`, `gradtest`, `ngradient`).

Supported Julia: **1.10+** (`Project.toml` `compat`; the README's "1.6 onwards" is stale).

## Architecture

### The differentiation pipeline

Everything funnels through `_pullback(ctx, f, args...)`, a **generated function** in `src/compiler/interface2.jl`. For a given call signature it does, in order:

1. **ChainRules first** (`src/compiler/chainrules.jl`): if a `ChainRulesCore.rrule` exists for the signature, use it. `has_chain_rrule` carefully handles `rrule` vs `no_rrule` opt-outs and attaches recompilation edges so newly-defined rules invalidate cached code.
2. **Source transform otherwise**: fetch the lowered IR of `f(args...)`, run the adjoint transform in `src/compiler/reverse.jl` (this builds *two* IR fragments — the instrumented forward pass and the backward/pullback pass), then `src/compiler/emit.jl` lowers them into runnable IR. The forward pass returns `(y, Pullback{Tuple{typeof(f),...}}(data))`.

`Pullback{S,T}` (defined in `interface.jl`) is a struct, because lowered IR can't hold a real closure. Calling it (`(j::Pullback)(Δ)`) hits a *second* generated function in `interface2.jl` that emits the backward code, using the captured `data`. The transform is recursive: the generated forward pass calls `_pullback` on every sub-call, so differentiation bottoms out at hand-written rules or ChainRules.

`@code_adjoint f(x)` shows the transformed IR; `Zygote.@code_ir f(x)` shows the input IR. See `docs/src/internals.md` for a worked example of what the transform produces.

### Layout

- **`src/compiler/`** — the AD engine. `interface.jl` (user-facing `gradient`, `withgradient`, `pullback`, `jacobian`, `hessian`, `diaghessian`, `Params`/`Grads`, `Context`); `interface2.jl` (the two generated functions); `reverse.jl` (the IR→IR adjoint transform, built on **IRTools**); `emit.jl` (lowering, stacks for control flow); `chainrules.jl` (rule lookup + ChainRules↔Zygote tangent conversion); `show.jl`.
- **`src/lib/`** — hand-written gradients for Base/stdlib, defined as `@adjoint` macros or direct `_pullback(cx::AContext, ::typeof(g), ...)` method overloads that short-circuit the source transform. Organized by domain: `array.jl`, `base.jl`, `broadcast.jl`, `number.jl`, `range.jl`, `forward.jl`, `literal_getproperty.jl` (struct field access), `logexpfunctions.jl`. **Most new gradient code goes here.**
- **`src/forward/`** — a separate, smaller **forward-mode** (pushforward) system, used e.g. for forward-over-reverse in `hessian`.
- **`src/tools/`** — `Buffer` (a mutable array type that *is* differentiable, the supported workaround for array mutation) and `builtins`.
- **`ext/`** — package extensions (Tracker, Colors, Distances, Atom) loaded via `weakdeps`.
- **ZygoteRules** (separate package) provides the `@adjoint`/`@adjoint!` macros and `_pullback` symbols that `src/lib` builds on.

### Context and implicit parameters

`Context{I} <: AContext` threads through every `_pullback` call and caches gradients of mutable objects in an `IdDict`. The type parameter `I` selects the gradient mode: `I=false` (default) is the modern explicit style (`gradient(f, args...)` returns a tuple); `I=true` enables the legacy **implicit** `Params`/`Grads` API (`gradient(() -> ..., Params([W, b]))`), which accumulates gradients of globally-referenced arrays. Implicit params are legacy — prefer explicit gradients in new code.

## Adding or changing gradients

- **Prefer a `ChainRulesCore.rrule`** when the rule is broadly useful and provider-agnostic — many rules now live in ChainRules.jl, not here.
- Use Zygote's `@adjoint` (from `src/lib/`) for Zygote-specific behavior or things tied to its internals (hooks, checkpointing, gradient reflection). Pattern: `@adjoint f(x) = f(x), ȳ -> (x̄,)` — the pullback returns one tangent **per argument**.
- For lower-level control, overload `_pullback(cx::AContext, ::typeof(f), args...)` directly (common throughout `src/lib`).
- Gradient w.r.t. the *function itself* is the first slot of `_pullback`'s output (usually `nothing`, but non-`nothing` for closures that capture data). `pullback`/`gradient` strip this slot via `tailmemaybe`.
- Useful primitives: `Zygote.hook(f, x)` (apply `f` to the gradient flowing back through `x`, `src/lib/utils.jl`), `Zygote.checkpointed(f, xs...)` (recompute in the backward pass to save memory, `src/lib/grad.jl`). To mark things non-differentiable use `ChainRulesCore.@non_differentiable` / `ChainRulesCore.ignore_derivatives` — Zygote's own `@nograd` and `dropgrad` are **deprecated** (see `src/deprecated.jl`) and just forward to those.

## Gotchas

- **Generated functions + editing.** Pullbacks are produced by generated functions, so editing `src/compiler/*` (the transform machinery) generally needs a **fresh Julia session** — Revise cannot reliably pick those up. Editing `src/lib` adjoints is friendlier, but already-compiled pullbacks for a signature may be stale until redefined/restarted.
- **No array mutation, no `try`/`catch`, no foreign calls.** These throw at differentiation time. The mutation workaround is `Buffer`; see `docs/src/limitations.md`.
- **Nested AD** (differentiating through `_pullback`/pullbacks) is supported but fragile — note the explicit `@adjoint` on `tailmemaybe` in `interface.jl` that exists solely to make nesting work. Tread carefully when touching tangent-unthunking or the interface wrappers.
- **Version-specific test breakage** is expected and encoded inline (`broken = VERSION >= v"1.12" && ...`) — Julia compiler internals shift between releases. Match this pattern rather than deleting tests when a case breaks on a new Julia.
- **Debugging a bad gradient:** use `_pullback` directly and recurse into sub-calls to isolate the offending function (recipe in `docs/src/internals.md`).
