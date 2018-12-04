using DataFrames
using LinearAlgebra

## Setting up numerical integration methods
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

# num_method_compare(a = 0.0, b = 10.0, f = (x -> 1/sqrt(2*pi) * exp(-x^2/2)))

### Problem 3 ###
num_method_compare(a = 1.0, b = 3.0, f = (x -> sqrt(x)*exp(-x)))

# │ Row │ no_intervals │ midpoint │ trapezoidal │ simpsons │
# ├─────┼──────────────┼──────────┼─────────────┼──────────┤
# │ 1   │ 4            │ 0.407157 │ 0.410757    │ 0.408357 │
# │ 2   │ 8            │ 0.408075 │ 0.408957    │ 0.408369 │
# │ 3   │ 16           │ 0.408297 │ 0.408516    │ 0.40837  │
# │ 4   │ 32           │ 0.408352 │ 0.408407    │ NaN      │
# │ 5   │ 64           │ 0.408366 │ 0.408379    │ NaN      │
# │ 6   │ 128          │ 0.408369 │ 0.408373    │ NaN      │
# │ 7   │ 256          │ 0.40837  │ 0.408371    │ NaN      │
# │ 8   │ 512          │ NaN      │ 0.40837     │ NaN      │

### Problem 4 Parts (i) & (iii) ###
num_method_compare(a = 0.0, b = 1.0, init_int = 2.0, f = (x -> (x^(5/2))/(1 + x^2)))

# │ Row │ no_intervals │ midpoint │ trapezoidal │ simpsons │
# ├─────┼──────────────┼──────────┼─────────────┼──────────┤
# │ 1   │ 2            │ 0.17059  │ 0.195711    │ 0.178964 │
# │ 2   │ 4            │ 0.177157 │ 0.183151    │ 0.179155 │
# │ 3   │ 8            │ 0.178678 │ 0.180154    │ 0.17917  │
# │ 4   │ 16           │ 0.179049 │ 0.179416    │ 0.179171 │
# │ 5   │ 32           │ 0.179141 │ 0.179232    │ 0.179171 │
# │ 6   │ 64           │ 0.179164 │ 0.179186    │ NaN      │
# │ 7   │ 128          │ 0.179169 │ 0.179175    │ NaN      │
# │ 8   │ 256          │ 0.179171 │ 0.179172    │ NaN      │
# │ 9   │ 512          │ 0.179171 │ 0.179171    │ NaN      │

### Problem 13 Parts (i) & (ii) ###
prob13_int1 = num_method_compare(a = 0.0, b = 1.0, init_int = 2.0, f = (x -> 0.05/(1 + exp(-1*(1+x)^2))), tol = 1e-6)[:simpsons]
prob13_int2 = num_method_compare(a = 0.0, b = 2.0, init_int = 2.0, f = (x -> 0.05/(1 + exp(-1*(1+x)^2))), tol = 1e-6)[:simpsons]
prob13_int3 = num_method_compare(a = 0.0, b = 3.0, init_int = 2.0, f = (x -> 0.05/(1 + exp(-1*(1+x)^2))), tol = 1e-8)[:simpsons]

filter!(!isnan, prob13_int1)
filter!(!isnan, prob13_int2)
filter!(!isnan, prob13_int3)

# Probably a better way to do this
prob13_int1 = prob13_int1[length(prob13_int1)]
prob13_int2 = prob13_int2[length(prob13_int2)]
prob13_int3 = prob13_int3[length(prob13_int3)]

# Part (i) answers
prob13_df1 = exp(-prob13_int1) # 0.9565953817710231
prob13_df2 = exp(-prob13_int2) # 0.9101276309729124
prob13_df3 = exp(-prob13_int3) # 0.8657410344688189

# Part(ii) answer
prob13_pt2 = 5*prob13_df1 + 5*prob13_df2 + 105*prob13_df3
# 100.23642368294566

#

## Functions to get bond Price, Duration, and Convexity
"""
Computes bond price.

Inputs:
	coupon = coupon rate in percent (i.e. 6.0 = 6% coupon),
	yield = yield in percent,
	time = time to maturity in years,
	freq = number of coupon payments per year

Output:
	B = bond price
"""
function PriceDurConv(;coupon::Float64, yield::Float64, time::Float64, freq::Float64) where T <:Dict{String,Float64}

	# Setting up cashflow vector
	cashflows = fill(coupon/freq, convert(Int64,ceil(time*freq)))

	# Adding face to last cash flow
	cashflows[length(cashflows)] += 100.0

	# Setting up time vector
	ts = collect(time:-1/freq:0)

	ts = ts[ts .> 0]

	ts = reverse(ts)

	# Price
	B = LinearAlgebra.dot(cashflows,exp.(-yield/100*ts))

	# Duration
	D = LinearAlgebra.dot(ts.*cashflows,
		exp.(-yield/100*ts))/B

	# Convexity
	C = LinearAlgebra.dot((ts.^2.0).*cashflows,
		exp.(-yield/100*ts))/B

	# Bringing it all together
	return Dict("Price" => B,
		"Duration" => D,
		"Convexity" => C)
end

#

## Problem 14 ###
PriceDurConv(coupon = 6.0, yield = 9.0, time = 30/12, freq = 2.0)
  # "Duration"  => 2.35242
  # "Convexity" => 5.73674
  # "Price"     => 92.9839

## Problem 15 ###
PriceDurConv(coupon = 8.0, yield = 7.0, time = 14/12, freq = 4.0)
  # "Duration"  => 1.11891
  # "Convexity" => 1.28571
  # "Price"     => 101.705

 