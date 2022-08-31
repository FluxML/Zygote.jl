using Documenter, Zygote


makedocs(
  sitename="Zygote",
  doctest = false,
  pages = [
        "Home" => "index.md",
        "Limitations" => "limitations.md",
        "Custom Adjoints" => "adjoints.md",
        "Utilities" => "utils.md",
        "Complex Differentiation" => "complex.md",
        "Profiling" => "profiling.md",
        "Internals" => "internals.md",
        "Glossary" => "glossary.md"],
  format = Documenter.HTML(
      prettyurls = haskey(ENV, "CI"),
      assets = ["assets/flux.css"],
      analytics = "UA-36890222-9"
  )
)

deploydocs(
    repo = "github.com/FluxML/Zygote.jl.git",
    target = "build",
    push_preview = true
)
