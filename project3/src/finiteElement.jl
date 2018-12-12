using Polynomials
using UnicodePlots
using LinearAlgebra
using FastGaussQuadrature
using RowEchelon

function LagrangeInterpolatingPolynomial(x::Array{<:Real,1}, j::Int)
    p = Poly([1.0])
    d = 1.0
    for k = 1:length(x)
        if k != j 
            pₖ = Poly([-x[k], 1.0])
            dₖ = (x[j] - x[k])
            p *= pₖ
            d *= dₖ
        end
    end
    return p / d
end

function ∂LagrangeInterpolatingPolynomial(x::Array{<:Real,1}, j::Int)
    p = LagrangeInterpolatingPolynomial(x, j)
    ∂p∂x = polyder(p, 1)
    return ∂p∂x
end

function piecewise(xx::Array{<:Real,1}, pieces::Array{<:Real,1}, lower::Poly{Float64}, upper::Poly{Float64})
    y = copy(xx)
    for i = 1:length(xx)
        x = xx[i]
        if x < pieces[1]
            y[i] = 0
        elseif x > pieces[1] && x < pieces[2]
            y[i] = lower(x)
        elseif x >= pieces[2] && x < pieces[3]
            y[i] = upper(x)
        else 
            y[i] = 0
        end
    end
    return y
end

struct basisPolynomial
    basisPolynomial() = new([0.0], 0, 0, 0, 0)
    basisPolynomial(s    ::Array{<:Real, 1}, 
                    p₀   ::Poly{<:Real}, 
                    ∂p₀∂x::Poly{<:Real}, 
                    p₁   ::Poly{<:Real}, 
                    ∂p₁∂x::Poly{<:Real}) = new(s, p₀, ∂p₀∂x, p₁, ∂p₁∂x)

    support::Array{<:Real, 1}
    p₀   ::Poly{<:Real}
    ∂p₀∂x::Poly{<:Real}
    p₁   ::Poly{<:Real}
    ∂p₁∂x::Poly{<:Real}
end

function piecewise(x::Array{<:Real,1}, basis::basisPolynomial)
    return piecewise(x, 
                     basis.support, 
                     basis.p₀,
                     basis.p₁)
end

function ∂piecewise(x::Array{<:Real,1}, basis::basisPolynomial)
    return piecewise(x, 
                     basis.support, 
                     basis.∂p₀∂x,
                     basis.∂p₁∂x)
end

function FEM_P2(N::Int)
    # In order to use P2 basis, we need the total number of points to be odd.
    @assert N >= 5
    @assert N%2 == 1

    nodes    = collect(range(0.0, stop=1.0, length=N))
    elements = [[j for j = i:i+2] for i = 1:2:N-2]

    # Generate basis of P2 functions
    basis = []

    for i = 1:length(elements)
        for j = 1:length(elements[i])
            p₀ = Poly([0.0])
            p₁ = Poly([0.0])
            ∂p₀∂x = Poly([0.0])
            ∂p₁∂x = Poly([0.0])

            if j == 1
                if i == 1
                    p₀    = Poly([0.0])
                    ∂p₀∂x = Poly([0.0])

                    ind = elements[1]
                    support = [nodes[ind[1]], nodes[ind[1]], nodes[ind[end]]]
                else 
                    n     = [nodes[elements[i-1][1]], nodes[elements[i-1][2]], nodes[elements[i-1][end]]]
                    p₀    = LagrangeInterpolatingPolynomial(n, length(elements[i-1]))
                    ∂p₀∂x = polyder(p₀)   
                end
                n     = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
                p₁    = LagrangeInterpolatingPolynomial(n, 1)
                ∂p₁∂x = polyder(p₁)

                if i != 1 
                    ind0 = elements[i-1]
                    ind1 = elements[i]
                    support = [nodes[ind0[1]], nodes[ind1[1]], nodes[ind1[end]]]
                end
            
            elseif j == length(elements[i])
                if i == length(elements)
                    p₁      = Poly([0.0])
                    ∂p₁∂x   = Poly([0.0]) 

                    ind     = elements[length(elements)]
                    support = [nodes[ind[1]], nodes[ind[end]], nodes[ind[end]]]
                else 
                    n     = [nodes[elements[i+1][1]], nodes[elements[i+1][2]], nodes[elements[i+1][end]]]
                    p₁    = LagrangeInterpolatingPolynomial(n, 1)
                    ∂p₁∂x = polyder(p₁)
                end
                n     = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
                p₀    = LagrangeInterpolatingPolynomial(n, length(elements[i]))
                ∂p₀∂x = polyder(p₀) 

                if i != length(elements)
                    ind0 = elements[i]
                    ind1 = elements[i+1]
                    support = [nodes[ind0[1]], nodes[ind1[1]], nodes[ind1[end]]]
                end
            
            else
                n     = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
                p₀    = LagrangeInterpolatingPolynomial(n, j)
                ∂p₀∂x = polyder(p₀) 

                p₁      = p₀
                ∂p₁∂x   = ∂p₀∂x
                support = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
            end

            if ! (i != 1 && j == 1)
                push!(basis, basisPolynomial(support, p₀, ∂p₀∂x, p₁, ∂p₁∂x))
            end
        end
    end

    function min(a::Array{<:Real,1}) 
        m = a[1]
        for i = 2:length(a)
            if a[i] < m
                m = a[i]
            end
        end
        return m
    end

    function max(a::Array{<:Real,1}) 
        m = a[1]
        for i = 2:length(a)
            if a[i] > m
                m = a[i]
            end
        end
        return m
    end

    # Integral points, weights
    x, ω = gausslegendre(100000)
    A = zeros(Float64, N, N)
    B = zeros(Float64, N)

    function f(x) 
        return -2.0
    end


    for i = 1:N
        bᵢ = basis[i]
        sᵢ = bᵢ.support

        a = sᵢ[1]
        b = sᵢ[end]

        uᵢf  = piecewise(x*(b-a)/2 .+ (a+b)/2, bᵢ) .* f(x*(b-a)/2 .+ (a+b)/2)
        ∫uᵢf = (b-a)/2 * sum(uᵢf .* ω)
        B[i] = ∫uᵢf 

        for j = i:N
            bⱼ = basis[j]
            sⱼ = bⱼ.support

            if sᵢ[end] < sⱼ[1]
                A[i,j] = 0.0
                A[j,i] = 0.0
            end

            a = min(sᵢ)
            b = max(sⱼ)
            ∂uᵢ∂uⱼ = ∂piecewise(x*(b-a)/2 .+ (a+b)/2, bᵢ) .* 
                     ∂piecewise(x*(b-a)/2 .+ (a+b)/2, bⱼ)
            ∫∂uᵢ∂uⱼ = (b-a)/2 * sum(∂uᵢ∂uⱼ .* ω)

            # Minus sign from the integration by parts in Galerkin method
            A[i,j] = -∫∂uᵢ∂uⱼ
            A[j,i] = -∫∂uᵢ∂uⱼ
        end
    end
    A[1,:]    .= 0.0
    A[end,:]  .= 0.0
    A[1,1]     = 1.0
    A[end,end] = 1.0
    B[1]   = 0.0
    B[end] = 0.0

    function showarray(array::Array{<:Number,<:Number})
        display(array)
        println("")
    end


    uₐₚₚ = A \ B
    #N = 100
    x = nodes #collect(range(0.0, stop=1.0, length=N))
    uₑₓ = x - x.^2

    function abs!(array::Array{<:Number, 1})
        for i = 1:length(array)
            array[i] = abs(array[i])
        end
        return array
    end

    #println(lineplot(nodes, uₐₚₚ))
    #println(lineplot(x, uₑₓ))
    println("N: ", N, "  e: ", sum(abs!(uₑₓ .- uₐₚₚ))/N)
end


for N = 5:10:101
    FEM_P2(N)
end


