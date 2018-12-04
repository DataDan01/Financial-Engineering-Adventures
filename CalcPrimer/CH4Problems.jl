using Distributions
using LinearAlgebra
using DataFrames

## Simulations off problems 3 to 6
"""
Computes analytical prices for vaniall European Call and Put options under Black-Scholes-Merton.

Inputs:
	s = Spot Price
	k = Strike Price
	r = Annualized risk free rate
	q = Annualized dividend yield, assumed to be continuous
	t = Time to maturity in year
	sd = Annualized volatility

Output:
	Dictionary with analytical Call and Put price
"""
function BSMprice(;s::Float64, k::Float64, r::Float64, q::Float64 = 0.0, t::Float64, sd::Float64) where T <:Dict{String,Float64}

	d1 = (log(s/k) + (r - q + sd^2/2)*t) / (sd*sqrt(t))

	d2 = d1 - sd*sqrt(t)

	Call_Price = s*exp(-q*t)*cdf(Normal(0.0,1.0), d1) - k*exp(-r*t)*cdf(Normal(0.0,1.0), d2)

	Put_Price = k*exp(-r*t)*cdf(Normal(0.0,1.0), -d2) - s*exp(-q*t)*cdf(Normal(0.0,1.0), -d1)

	return Dict("Call_Price" => Call_Price,
		"Put_Price" => Put_Price)
end

##

"""
Numerically estimates Call/Put price under Cox-Ross-Rubinstein parameterization of a binomial tree.

Inputs:
	s = Spot Price
	k = Strike Price
	r = Annualized risk free rate
	q = Annualized dividend yield, assumed to be continuous
	t = Time to maturity in year
	sd = Annualized volatility
	num_incs = Number of discrete time increments
	num_sims = Number of simlations to run for estimate

Output:
	Dictionary with estimated Call and Put price

"""
function CRR_price_est(;s::Float64, k::Float64, r::Float64, t::Float64, sd::Float64, method::String = "P5", num_incs::Int64 = 1000, num_sims::Int64 = 100000) where T <:Dict{String,Float64}

	dt = t/num_incs

	# Parameterization from problem 5
	if method == "P5"

		A = 1/2*(exp(-r*dt) + exp((r + sd^2)*dt))

		u = A + sqrt(A^2 - 1)
		d = A - sqrt(A^2 - 1)

		p = (exp(r*dt) - d)/(u - d)
	end

	#

	# Parameterization from problem 6
	if method == "P6"

		u = exp(r*dt)*(1 + sqrt(exp(sd^2*dt) - 1))
		d = exp(r*dt)*(1 - sqrt(exp(sd^2*dt) - 1))

		p = (1/2)
	end

	#

	mean_payoff_call = 0.0
	mean_payoff_put = 0.0

	for i = 1:num_sims

		up_moves = rand(Binomial(num_incs,p), 1)[1]
		down_moves = num_incs - up_moves

		net_moves = up_moves - down_moves

		# May run into numerical stability issues
		end_price = (s * u^up_moves * d^down_moves) 

		mean_payoff_call += max(end_price - k, 0) / num_sims
		mean_payoff_put += max(k - end_price, 0) / num_sims
	end
	#

	return Dict("Call_Price_Est" => (mean_payoff_call * exp(-r*t)),
		"Put_Price_Est" => (mean_payoff_put * exp(-r*t)))
end


BSMprice(s = 100.0, k = 100.0, r = 0.03, t = 3.0, sd = 0.15)
CRR_price_est(s = 100.0, k = 100.0, r = 0.03, t = 3.0, sd = 0.15)
CRR_price_est(s = 100.0, k = 100.0, r = 0.03, t = 3.0, sd = 0.15, method = "P6")

# Dict{String,Float64} with 2 entries:
#   "Call_Price" => 14.7782
#   "Put_Price"  => 6.17132

# Dict{String,Float64} with 2 entries:
#   "Put_Price_Est"  => 6.18454
#   "Call_Price_Est" => 14.725

# Dict{String,Float64} with 2 entries:
#   "Put_Price_Est"  => 6.26414
#   "Call_Price_Est" => 14.6658

## Problem 7
s = 50
q = 0.02
u = 0.08
sd = 0.3
r = 0.05

# Part (i) & (ii)
t = [1/24; 1/12; 2/12; 6/12; 1.0]

pct_99 = quantile(Normal(0,1), 1 - 0.01/2)
pct_95 = quantile(Normal(0,1), 1 - 0.05/2)

DataFrame(
	# Regular
	Lower_95 = (s .* exp. ((u - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* -pct_95)), 
	(Upper_95 = s .* exp. ((u - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* pct_95))
	,
	Lower_99 = (s .* exp. ((u - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* -pct_99)), 
	(Upper_99 = s .* exp. ((u - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* pct_99)),
	# Risk Neutral
	RN_Lower_95 = (s .* exp. ((r - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* -pct_95)), 
	(RN_Upper_95 = s .* exp. ((r - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* pct_95))
	,
	RN_Lower_99 = (s .* exp. ((r - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* -pct_99)), 
	(RN_Upper_99 = s .* exp. ((r - q - sd^2/2) .* t .+ sd .* sqrt.(t) .* pct_99))
	)

## Problem 8
p = 18/(18+18+2)
games = 100

# Explicit solution
1 - cdf(Binomial(games, p), 54) # 0.07668925524900927
cdf(Binomial(games, p), 45) # 0.3547937442587473

## Problem 9
s = 60
k = 55
t = 4/12
u = 0.1
sd = 0.3
r = 0.05
q = 0

# Part (i)
cdf(Normal(0,1), -(log(s/k) + (u - q - sd^2/2)*t)/(sd*sqrt(t)))
# 0.27152477509629924

# Part (ii)
cdf(Normal(0,1), -(log(s/k) + (r - q - sd^2/2)*t)/(sd*sqrt(t)))
# 0.30433148036687085

## Problem 11

# Setting up numerical integration functions, copied from CH2

"""
Computes the Midpoint numerical integration method.

Inputs:
	a = Lower endpoint in integral,
	b = Upper endpoint in integral,
	n = Number of partitions,
	f = The function to integrate

Output:
	I_midpoint = Result of approximate numerical integration
"""
function midpoint(;a::Float64, b::Float64, n::Float64, f::Function) where T <:Float64

	n_int = convert(UInt64, n) # Speeds things up

	h = (b-a)/n

	I_midpoint = 0.0

	for i = 1:n_int
		I_midpoint += f(a + (i-1/2)*h)
	end

	I_midpoint *= h

	return I_midpoint
end

# 

"""
Computes the Trapezoidal numerical integration method.

Inputs:
	a = Lower endpoint in integral,
	b = Upper endpoint in integral,
	n = Number of partitions,
	f = The function to integrate

Output:
	I_trap = Result of approximate numerical integration
"""
function trapezoidal(;a::Float64, b::Float64, n::Float64, f::Function) where T <:Float64

	n_int = convert(UInt64, n) # Speeds things up

	h = (b-a)/n

	I_trap = f(a)/2 + f(b)/2

	for i = 1:(n_int-1)
		I_trap += f(a + i*h)
	end

	I_trap *= h

	return I_trap
end

#

"""
Computes the Simpson's numerical integration method.

Inputs:
	a = Lower endpoint in integral,
	b = Upper endpoint in integral,
	n = Number of partitions,
	f = The function to integrate

Output:
	I_simp = Result of approximate numerical integration
"""
function simpsons(;a::Float64, b::Float64, n::Float64, f::Function) where T <:Float64

	n_int = convert(UInt64, n) # Speeds things up

	h = (b-a)/n

	I_simp = f(a)/6 + f(b)/6

	for i = 1:n_int
		if i <= (n_int-1)
			I_simp += f(a + i*h)/3
		end

		I_simp += 2*f(a + (i-1/2)*h)/3
	end

	I_simp *= h

	return I_simp
end

#

# midpoint(a = 1.0, b = 3.0, n = 10000.0, f = (x -> 1/sqrt(2*pi) * exp(-x^2/2)))
# trapezoidal(a = 1.0, b = 3.0, n = 10000.0, f = (x -> 1/sqrt(2*pi) * exp(-x^2/2)))
# simpsons(a = 1.0, b = 3.0, n = 10000.0, f = (x -> 1/sqrt(2*pi) * exp(-x^2/2)))

#
"""
Applies various numerical integration methods and compares convergence. 

Inputs:
	a = Lower endpoint in integral,
	b = Upper endpoint in integral,
	init_int = Initial number of intervals,
	tol = Tolerance - stopping rule for mesh refinements,
	max_iter = Maximum number of mesh refinements,
	int_methods = Integration methods to apply
	f = Function to integrate

Output:
	result = DataFrame comparing convergence across mesh refinements
"""
function num_method_compare(;a::Float64, b::Float64, init_int::Float64 = 4.0, tol = 1e-6::Float64, 
	max_iter::Int64 = 20, int_methods::Array{Function,1} = [midpoint, trapezoidal, simpsons], f::Function) where T <:DataFrame

	# Setting up the return object
	result = DataFrame(no_intervals = [2.0^i * init_int for i = 0:max_iter])

	# Looping through methods
	for method in int_methods

		# Initializing new column
		result[:temp] = NaN

		# First integration run
		result[1,:temp] = method(a = a, b = b, n = result[1,:no_intervals], f = f)

		# Subsequent integration runs
		for i in 2:max_iter

			result[i,:temp] = method(a = a, b = b, n = result[i,:no_intervals], f = f)

			# Stop running if done improvement within tolerance
			if abs(result[i,:temp] - result[i-1,:temp]) < tol
				break
			end

			if(i == max_iter)
				@warn("Did not converge to desired tolerance before maximum number of iterations!\nTry lowering the tolerance or increasing the maximum number of iterations.")
			end
		end

		# Renaming temp column to match numerical method name
		rename!(result, :temp =>  Symbol(string(method)))

	end

	# Cleaning up DataFrame
	result[:no_intervals] = Array{Int64,1}(result[:no_intervals])

	# Filtering down table to maximum iteration of any method
	filt_inx = colwise(x -> max_iter - sum(isnan.(x)) + 1, result)
	filt_inx = deleteat!(filt_inx, 1)
	filt_inx = maximum(filt_inx)

	result = result[1:filt_inx,:]

	return result
end


# Setting up problem variables
t = 1/2
k = 50.0
q = 0.02
r = 0.04

s = 45.0
sd = 0.25

d2 = (log(s/k) + (r-q-sd^2/2)*t)/(sd*sqrt(t))



BSMprice(s = s, k = k, r = r, q = q, t = t, sd = sd) # 5.99018

num_method_compare(
	a = -5.0, 
	b = -d2, 
	init_int = 4.0, 
	tol = 1e-6, 
	max_iter = 20, 
	int_methods = [midpoint, trapezoidal, simpsons], 
	f = (x -> exp(-r*t)/sqrt(2*pi) * (k - s*exp((r-q-sd^2/2)*t + sd*sqrt(t)*x)) * exp(-x^2/2)
))

# Very close to BSM value

#=12×4 DataFrame
│ Row │ no_intervals │ midpoint │ trapezoidal │ simpsons │
│     │ Int64        │ Float64  │ Float64     │ Float64  │
├─────┼──────────────┼──────────┼─────────────┼──────────┤
│ 1   │ 4            │ 6.25911  │ 5.48287     │ 6.00036  │
│ 2   │ 8            │ 6.05056  │ 5.87099     │ 5.9907   │
│ 3   │ 16           │ 6.00491  │ 5.96077     │ 5.9902   │
│ 4   │ 32           │ 5.99383  │ 5.98284     │ 5.99017  │
│ 5   │ 64           │ 5.99108  │ 5.98834     │ 5.99017  │
│ 6   │ 128          │ 5.9904   │ 5.98971     │ 5.99017  │
│ 7   │ 256          │ 5.99022  │ 5.99005     │ NaN      │
│ 8   │ 512          │ 5.99018  │ 5.99014     │ NaN      │
│ 9   │ 1024         │ 5.99017  │ 5.99016     │ NaN      │
│ 10  │ 2048         │ 5.99017  │ 5.99016     │ NaN      │
│ 11  │ 4096         │ 5.99017  │ 5.99017     │ NaN      │
│ 12  │ 8192         │ NaN      │ 5.99017     │ NaN      │=#

