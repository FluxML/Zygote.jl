using Documenter, Zygote

makedocs(
  sitename="Zygote",
  analytics = "UA-36890222-9",
  pages = [
        "Home" => "index.md",
        "Custom Adjoints" => "adjoints.md",
        "Utilities" => "utils.md",
        "Complex Differentiation" => "complex.md",
        "Flux" => "flux.md",
        "Profiling" => "profiling.md",
        "Internals" => "internals.md",
        "Glossary" => "glossary.md"],
  format = Documenter.HTML(prettyurls = haskey(ENV, "CI")))

deploydocs(
    repo = "github.com/FluxML/Zygote.jl.git",
)
