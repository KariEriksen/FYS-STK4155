using Polynomials

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
