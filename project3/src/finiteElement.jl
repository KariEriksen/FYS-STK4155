using Polynomials
using UnicodePlots
using LinearAlgebra
using FastGaussQuadrature
using RowEchelon

include("lagrangeInterpolation.jl")
include("FEMBasis.jl")
include("FEMSolver.jl")


function f(x) 
    return sin.(π.*x)
end

function uₑ(x, t)
    return sin.(π.*x).*exp(-π^2 .* t)
end

function FEM_P2(N::Integer)
    nodes, elements, basis = generateP2Basis(N)

    K, M, f⁰ = setupMatrices(nodes, elements, basis, f)
    #u⁰ = projectinitial(nodes, f)
    u⁰ = projectinitial(M, f⁰)

    α  = 1 # Parameter in the diffusion equation
    t  = 0
    Nₜ = 1000
    Δt = 0.01
    β  = α * Δt

    uⁿ = u⁰
    for tᵢ = 1:Nₜ
        t = t + Δt 
        uⁿ⁺¹ = iterate_forward(K, M, uⁿ, β)
        uⁿ = uⁿ⁺¹

        if true #tᵢ % 10 == 0
            println(lineplot(nodes, uₑ.(nodes,t), ylim=[0,1]))
            println(lineplot(nodes, uⁿ, ylim=[0,1]))
            println(sum(abs.(uₑ(nodes,t).-uⁿ))/N)
            sleep(0.5)
        end
    end
end



FEM_P2(5)


