using BruteForceAllocationSolver
using Documenter

makedocs(;
    modules=[BruteForceAllocationSolver],
    authors="Sliem el Ela",
    repo="https://github.com/sliemelela/BruteForceAllocationSolver.jl",
    sitename="BruteForceAllocationSolver.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sliemelela.github.io/BruteForceAllocationSolver.jl",
        edit_link="main",
        assets=String[],
    ),
    checkdocs = :exports,
    pages=[
        "Home" => "index.md",
        "Theory (Background)" => "theory.md",
        "Tutorial" => "tutorial.md",
        "Tests" => "test.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/sliemelela/BruteForceAllocationSolver.jl",
    devbranch="main",
)