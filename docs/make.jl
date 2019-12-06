using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Zygote

makedocs(
  sitename="Zygote",
  pages = [
        "Home" => "index.md",
        "Custom Adjoints" => "adjoints.md",
        "Utilities" => "utils.md",
        "Complex Differentiation" => "complex.md",
        "Profiling" => "profiling.md",
        "Internals" => "internals.md",
        "Glossary" => "glossary.md"],
  format = Documenter.HTML(prettyurls = haskey(ENV, "CI"), analytics = "UA-36890222-9"))

deploydocs(
    repo = "github.com/FluxML/Zygote.jl.git",
)
