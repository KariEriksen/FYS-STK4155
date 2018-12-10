using Polynomials
using UnicodePlots

function LagrangeInterpolatingPolynomial(x::Array, j::Int)
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

function LagrangeInterpolatingPolynomialDerivative(x::Array, j::Int)
    p = LagrangeInterpolatingPolynomial(x, j)
    dpdx = polyder(p, 1)
    return dpdx
end





M = 3
x = [0,0.5,1]
println(x)

for j = 1:M
    p = LagrangeInterpolatingPolynomial(x, j)
    println(p)
    println(p(x))
    println("")
end

N  = 100
xx = range(0.0, stop=1.0, length=N)

for j = 1:M
    p = LagrangeInterpolatingPolynomial(x, j)
    plot = lineplot(xx, p(xx))
    println(plot)
end