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

    #K, M, f⁰ = setupmatrices_gaussianquadrature(nodes, elements, basis, f)
    K, M, f⁰ = setupmatrices_analytical(nodes, elements, basis, f)
    #u⁰ = projectinitial(nodes, f)
    u⁰ = projectinitial(M, f⁰)

    println("M")
    display(M)
    println("K")
    display(K)
    println(nodes)
    exit()
    

    α  = 1 # Parameter in the diffusion equation
    t₀ = 0
    t₁ = 0.1 
    Nₜ = 100
    Δt = (t₁ - t₀) / Nₜ
    β  = α * Δt
    ε = []

    uⁿ = u⁰
    t = t₀
    for tᵢ = 1:Nₜ
        t = t + Δt 
        #uⁿ⁺¹ = iterate_forward(K, M, uⁿ, β)
        uⁿ⁺¹ = iterate_backward(K, M, uⁿ, β)
        uⁿ = uⁿ⁺¹

        if tᵢ % 10 == 0
            #err = sum(abs.(uₑ(nodes,t).-uⁿ))/N
            #push!(ε, err)
            #println(lineplot(nodes, uₑ.(nodes,t), ylim=[0,1]))
            #println(lineplot(nodes, uⁿ, ylim=[0,1]))
            #println(lineplot(nodes[2:end-1], log10.(1e-10.+abs.(uₑ(nodes[2:end-1],t) .- uⁿ[2:end-1])), ylim=[-10, 1]))
            println(sum(abs.(uₑ(nodes,t).-uⁿ))/N)
            #println(lineplot(log10.(ε)))
            #sleep(0.05)
        end
    end
end



FEM_P2(11)


