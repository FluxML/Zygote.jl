using Zygote
using ParallelTestRunner
using CUDA

run_cuda = !haskey(ENV, "GITHUB_ACTION") && CUDA.has_cuda()
run_cuda || @warn "CUDA not found - Skipping CUDA Tests"

# Auto-discover all `.jl` files in `test/` (except `runtests.jl`).
testsuite = find_tests(@__DIR__)

# Remove helper files that are not standalone test files:
# - the gradcheck test setup module, included by the `gradcheck_*` files
# - the `lib/` files, included by `lib.jl`
delete!(testsuite, "gradcheck_testsetup")
delete!(testsuite, "lib/array")
delete!(testsuite, "lib/base")
delete!(testsuite, "lib/lib")
delete!(testsuite, "lib/number")

# The PythonCall tests require a configured Python environment, so they are
# disabled by default.
delete!(testsuite, "python")

if run_cuda
    # When a GPU is available, only run the CUDA tests.
    filter!(((k, _),) -> k == "cuda", testsuite)
else
    delete!(testsuite, "cuda")
end

# Code run in every test subprocess before the test file is included.
init_code = quote
    using Zygote
end

runtests(Zygote, ARGS; testsuite, init_code)
