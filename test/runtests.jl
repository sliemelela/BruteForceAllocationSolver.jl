using BruteForceAllocationSolver
using Test

@testset "BruteForceAllocationSolver.jl Tests" begin


    include("terminal_wealth_test.jl")
    include("merton_test.jl")
    include("stochastic_rate_test.jl")
    include("stock_bond_test.jl")
end

