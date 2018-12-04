using LinearAlgebra

## Setting up LU methods
"""
Returns the LU factorization with or without pivoting.

Inputs:
	A = [n x n] matrix, Float64

Outputs: (All Float64)
	P = [n x n] permutation matrix
	L = [n x n] lower triangular matrix with 1s on the Diagonal
	U = [n x n] upper triangular matrix
"""
function LU_fact(;A::Array{Float64,2}, with_pivot::Bool) where T <:Array{Float64,2}

	# Make a local copy to avoid modifying global matrix
	A = copy(A)

	# Setting up outputs
	n = size(A)[1]

	if with_pivot == true
		P = zeros(Float64,size(A)) + Diagonal(ones(n))
	end

	U = zeros(Float64,size(A))
	L = zeros(Float64,size(A)) + Diagonal(ones(n))

	for i = 1:(n-1)

		if with_pivot == true
			# First index of the largest absolute value in first column
			i_max = findmax(abs.(A[i:n,i]))[2][1] + i - 1
			vv = A[i,:]
			A[i,:] = A[i_max,:]
			A[i_max,:] = vv

			cc = P[i,:]
			P[i,:] = P[i_max,:]
			P[i_max,:] = cc

			vv = cc = Nothing
		end

		if with_pivot == true
			# Permute L matrix when past the first step
			if i > 1
				ww = L[i,1:(i-1)]
				L[i,1:(i-1)] = L[i_max,1:(i-1)]
				L[i_max,1:(i-1)] = ww
				ww = Nothing
			end
		end

		# Update lower entries of U and L
		L[i:n,i] = A[i:n,i]/A[i,i]
		U[i,i:n] = A[i,i:n]

		A[(i+1):n,(i+1):n] = A[(i+1):n,(i+1):n] - 
			L[(i+1):n,i]*transpose(U[i,(i+1):n])
	end

	# Final update
	L[n,n] = 1
	U[n,n] = A[n,n]

	# Returning outputs
	if with_pivot == true
		output = Dict("P" => P, "L" => L, "U" => U)
	end

	if with_pivot == false
		output = Dict("L" => L, "U" => U)
	end

	return output
end

#

"""
Performs forward substitution to solve a linear system with a lower triangular nonsingular matrix.

Inputs:
	L = [n x n] lower triangular nonsingular matrix
	b = [n x 1] result vector from linear system

Output:
	x = [n x 1] column vector with the inputs to the linear system
"""
function forward_subst(;L::Array{Float64,2}, b::Array{Float64,1}) where T <:Array{Float64,1}

	n = size(b)[1]

	x = zeros(n)

	# First pass
	x[1] = b[1]/L[1,1]

	# Subsequent passes
	for j = 2:(size(b)[1])
		x[j] = (b[j] - transpose(L[j,1:(j-1)])*(x[1:(j-1)]))/L[j,j]
	end

	return x
end

#

"""
Performs backward substitution to solve a linear system with a upper triangular nonsingular matrix.

Inputs:
	U = [n x n] upper triangular nonsingular matrix
	b = [n x 1] result vector from linear system

Output:
	x = [n x 1] column vector with the inputs to the linear system
"""
function backward_subst(;U::Array{Float64,2}, b::Array{Float64,1}) where T <:Array{Float64,1}

	n = size(b)[1]

	x = zeros(n)

	# First pass
	x[n] = b[n]/U[n,n]

	# Subsequent passes
	for j = (n-1):-1:1
		x[j] = (b[j] - transpose(U[j,(j+1):n])*(x[(j+1):n]))/U[j,j]
	end

	return x
end

# 
"""
Solves a linear system using the LU decomposition of a matrix (with pivoting)

Inputs:
	P, L, U = [n x n] outputs from LU_fact(A, with_pivot = true),
	b = [n x 1] output column vector

Output:
	x = [n x 1] linear system input vector
"""
function LU_linear_solver(;P::Array{Float64,2}, L::Array{Float64,2}, U::Array{Float64,2}, b::Array{Float64,1}) where T <:Array{Float64,1}

	P_b = P*b

	y = forward_subst(L = L, b = P_b)
	x = backward_subst(U = U, b = y)

	return x
end

#

"""
Finds the inverse of A using LU factorization with row pivoting.

Inputs:
	P, L, U = [n x n] outputs from LU_fact(A, with_pivot = true)

Output:
	A_inv = [n x n] inverse matrix where A*A_inv = I
"""
function find_inverse_LU(;P::Array{Float64,2}, L::Array{Float64,2}, U::Array{Float64,2}) where T <:Array{Float64,2}

	n = size(P)[1]

	A_inv = zeros(size(P))

	for k = 1:n
		y = forward_subst(L = L, b = P[:,k])
		A_inv[:,k] = backward_subst(U = U, b = y)
	end

	return A_inv
end

#

"""
Fits a Natural Cubic Spline to observed zero rates

Inputs:
	x = Vector of times
	v = Vector of zero rates
	time_0_rate = r(0,0)

Outputs:
	params = to be fed into cubic_spline
"""
create_spline_params = function(;x::Array{Float64,1}, v::Array{Float64,1}, time_0_rate::Float64) where T <:Array{Float64,1}

	n = size(v)[1]

	i = collect(1:n)
	i_c = collect(2:n)

	# Setting up `b` column vector
	b = zeros(4*n)

	b[(4 .* i_c) .- 2,:] = v[i_c .- 1,:]
	b[(4 .* i) .- 1,:] = v[i,:]

	i = i_c = Nothing

	# Including the r(0,0) `overnight` if it is given
	# This corresponds to v_0
	b[2] = time_0_rate

	# Setting up linear system `M`
	M = zeros(4*n,4*n)

	M[1,3] = 2 
	M[1,4] = 0

	M[4*n, 4*n-1] = 2
	M[4*n,4*n] = 6*x[n]

	for i = 1:n

		M[4*i-2,4*i-3] = 1

		if i > 1
			M[4*i-2,4*i-2] = x[i-1]

			M[4*i-2,4*i-1] = x[i-1]^2

			M[4*i-2,4*i] = x[i-1]^3
		end

		M[4*i-1,4*i-3] = 1

		M[4*i-1,4*i-2] = x[i]

		M[4*i-1,4*i-1] = x[i]^2	

		M[4*i-1,4*i] = x[i]^3


		if i < n
			M[4*i,4*i-2] = 1

			M[4*i,4*i-1] = 2*x[i]

			M[4*i,4*i] = 3*x[i]^2

			M[4*i,4*i+2] = -1

			M[4*i,4*i+3] = -2*x[i]

			M[4*i,4*i+4] = -3*x[i]^2

			M[4*i+1,4*i-1] = 2

			M[4*i+1,4*i] = 6*x[i]

			M[4*i+1,4*i+3] = -2

			M[4*i+1,4*i+4] = -6*x[i]
		end
	end

	##

	# First, get the LU factors of M
	fact_of_M = LU_fact(A = M, with_pivot = true)

	# Solve the linear system
	params = LU_linear_solver(
		P = fact_of_M["P"],
		L = fact_of_M["L"],
		U = fact_of_M["U"],
		b = b
	)

	return params
end

#

"""
Computes the value of the zero rate curve at a particular time. Zero rate curve parameters come from create_spline_params.

Inputs:
	t = Point on the zero rate curve
	x = Vector of time points from the original fitting procedure
	params = Result of create_spline_params

Output:
	zero_rate = Interpolated zero rate from curve
"""
cubic_spline = function(;t::Float64, x::Array{Float64,1}, params::Array{Float64,1}) where T <:Float64

	# Finding which function to apply
	inx = (t .>= [-Inf; x]) .* (t .<= [x; Inf])

	inx = findfirst(inx)
	
	# Setting up parameter values
	a_par = params[4*inx - 3]
	b_par = params[4*inx - 2]
	c_par = params[4*inx - 1]
	d_par = params[4*inx]

	# Applying function
	zero_rate = (a_par + b_par*t + c_par*t^2 + d_par*t^3)

	return zero_rate
end


#

## Problem 1 ##
L_1 = [1 0 0 0; -1 2 0 0; 2 -2 3 0; 2 2 -3 4.0]
U_1 = [2 -1 0 1; 0 -1/2 1/2 0; 0 0 1/3 -1/3; 0 0 0 -1/4]

L_2 = [1 0 0 0; -1 1 0 0; 2 -1 1 0; 2 1 -1 1.0]
U_2 = [2 -1 0 1; 0 -1 1 0; 0 0 1 -1; 0 0 0 -1.0]

sum(L_1*U_1 - L_2*U_2) # 0.0

sum(L_2 * Diagonal(collect(1:4.0)) - L_1) # 0.0

sum(Diagonal(1 ./ collect(1:4.0)) * U_2 - U_1) # 0.0

L_1 = U_1 = L_2 = U_2 = Nothing

## Problem 3 ##
A = [2 -1 0 1; -2 0 1 -1; 4 -1 0 1; 4 -3 0 2.0]

# Part (i)
fact_of_A = LU_fact(A = A, with_pivot = true)

fact_of_A["P"]
# 4×4 Array{Float64,2}:
# 0.0  0.0  1.0  0.0
# 0.0  0.0  0.0  1.0
# 0.0  1.0  0.0  0.0
# 1.0  0.0  0.0  0.0

fact_of_A["L"]
# 4×4 Array{Float64,2}:
#  1.0  0.0   0.0  0.0
#  1.0  1.0   0.0  0.0
# -0.5  0.25  1.0  0.0
#  0.5  0.25  0.0  1.0

fact_of_A["U"]
# 4×4 Array{Float64,2}:
# 4.0  -1.0  0.0   1.0 
# 0.0  -2.0  0.0   1.0 
# 0.0   0.0  1.0  -0.75
# 0.0   0.0  0.0   0.25

# Part(ii)
b = [3; -1; 0; 2.0]

LU_linear_solver(
	P = fact_of_A["P"],
	L = fact_of_A["L"],
	U = fact_of_A["U"],
	b = b)

# 4-element Array{Float64,1}:
# -1.5
#  4.0
#  6.0
# 10.0

# Part (iii)
A_inv = find_inverse_LU(
	P = fact_of_A["P"],
	L = fact_of_A["L"],
	U = fact_of_A["U"])

# 4×4 Array{Float64,2}:
#  -0.5   0.0   0.5   0.0
#   2.0  -0.0  -0.0  -1.0
#   3.0   1.0   0.0  -1.0
#   4.0   0.0  -1.0  -1.0

sum(A_inv * A - Diagonal(ones(size(A)[1]))) # 0.0

A = A_inv = fact_of_A = b = Nothing

## Problem 4 ##

A = [2 -1 3 -1; 1 0 -2 -4; 3 1 1 -2; -4 1 0 2.0]

b = [-1; 0; 1; 2.0]

# Part (i)
fact_of_A = LU_fact(A = A, with_pivot = true)

fact_of_A["P"]
# 4×4 Array{Float64,2}:
#  0.0  0.0  0.0  1.0
#  0.0  0.0  1.0  0.0
#  1.0  0.0  0.0  0.0
#  0.0  1.0  0.0  0.0

fact_of_A["L"]
# 4×4 Array{Float64,2}:
#  1.0    0.0        0.0       0.0
# -0.75   1.0        0.0       0.0
# -0.5   -0.285714   1.0       0.0
# -0.25   0.142857  -0.652174  1.0

fact_of_A["U"]
# 4×4 Array{Float64,2}:
# -4.0  1.0   0.0       2.0     
# 0.0  1.75  1.0      -0.5     
# 0.0  0.0   3.28571  -0.142857
# 0.0  0.0   0.0      -3.52174 

round(sum(fact_of_A["P"]*A - fact_of_A["L"]*fact_of_A["U"]),
		digits = 2) # -0.0

# Part (ii)
x = LU_linear_solver(
	P = fact_of_A["P"],
	L = fact_of_A["L"],
	U = fact_of_A["U"],
	b = b)

# 4-element Array{Float64,1}:
# -0.2716049382716049 
#  1.2592592592592593 
#  0.20987654320987653
# -0.1728395061728395 

# Part (iii)
P_1 = [1 0 0 0; 0 0 0 1; 0 1 0 0; 0 0 1 0.0;]
P_2 = [0 1 0 0; 0 0 0 1; 1 0 0 0; 0 0 1 0.0;]

L_1 = [1 0 0 0; 0 1 0 0; -0.6667 -0.5833 1 0; 0.3333 -0.5833 0.1429 1]
U_1 = [3 2 -1 -1; 0 -4 2 1; 0 0 -3.5 -0.0833; 0 0 0 1.9286]

round(sum(P_1 * A * P_2 - L_1 * U_1), digits = 2) # 0.0

# Part (iv)
y = forward_subst(L = L_1, b = (P_1*b))

x_1 = backward_subst(U = U_1, b = y)

round(sum(x - P_2*x_1), digits = 2) # 0.0

#

A = A_inv = fact_of_A = b = x = x_1 = Nothing

## Problem 5 ##

A = [1 0 0 0 1; -1 1 0 0 1; -1 -1 1 0 1; -1 -1 -1 1 1; -1 -1 -1 -1 1.0]

fact_of_A = LU_fact(A = A, with_pivot = false)

fact_of_A["L"]
# 5×5 Array{Float64,2}:
#  1.0   0.0   0.0   0.0  0.0
# -1.0   1.0   0.0   0.0  0.0
# -1.0  -1.0   1.0   0.0  0.0
# -1.0  -1.0  -1.0   1.0  0.0
# -1.0  -1.0  -1.0  -1.0  1.0

fact_of_A["U"]
# 5×5 Array{Float64,2}:
# 1.0  0.0  0.0  0.0   1.0
# 0.0  1.0  0.0  0.0   2.0
# 0.0  0.0  1.0  0.0   4.0
# 0.0  0.0  0.0  1.0   8.0
# 0.0  0.0  0.0  0.0  16.0

A - fact_of_A["L"]*fact_of_A["U"]

A = fact_of_A = nothing

## Problem 6 ##
A = [2 -1 1; -2 1 3; 4 0 -1.0]

# Part (i)
det([2 -1; -2 1.0]) # 0.0

## Part (ii)

fact_of_A = LU_fact(A = A, with_pivot = false)

fact_of_A["L"]
# 3×3 Array{Float64,2}:
#  1.0  0.0   0.0
# -1.0  NaN   0.0
#  2.0  Inf   1.0

fact_of_A["U"]
# 3×3 Array{Float64,2}:
# 2.0  -1.0     1.0
# 0.0   0.0     4.0
# 0.0   0.0  -Inf

# Part (iv)
det(A) # -16.0 != 0

fact_of_A = LU_fact(A = A, with_pivot = true)

fact_of_A["P"]
# 3×3 Array{Float64,2}:
# 0.0  0.0  1.0
# 0.0  1.0  0.0
# 1.0  0.0  0.0

fact_of_A["L"]
# 3×3 Array{Float64,2}:
# 1.0   0.0  0.0
#-0.5   1.0  0.0
# 0.5  -1.0  1.0

fact_of_A["U"]
# 3×3 Array{Float64,2}:
# 4.0  0.0  -1.0
# 0.0  1.0   2.5
# 0.0  0.0   4.0

fact_of_A = Nothing

## Problem 12
# Parts (i - iii)

# Note: do not weigh the cashflows by the 
# fraction of the year in the matrix
M = [
	1.5 101.5 0 0;
	2 2 102 0;
	0 6 0 106;
	2.5 2.5 2.5 102.5;
]

b = [101.30; 102.95; 107.35; 105.45]

# First get LU fact with pivot
fact_of_M = LU_fact(A = M, with_pivot = true)

# Then solve with LU solver
LU_linear_solver(
	P = fact_of_M["P"],
	L = fact_of_M["L"],
	U = fact_of_M["U"],
	b = b
)

# 4-element Array{Float64,1}:
# 0.9860402427758516
# 0.9834575333579924
# 0.9706961220365912
# 0.9570684415080382

# Problem 13
# Parts (i - ii)

M = [
	100 0 0 0;
	1.5 101.5 0 0;
	2.5 2.5 102.5 0;
	1.5 1.5 1.5 101.5
]

b = [98.5; 101; 102; 103.5]

# First get LU fact with pivot
fact_of_M = LU_fact(A = M, with_pivot = true)

# Then solve with LU solver
LU_linear_solver(
	P = fact_of_M["P"],
	L = fact_of_M["L"],
	U = fact_of_M["U"],
	b = b
)

# 4-element Array{Float64,1}:
# 0.985
# 0.9805172413793103
# 0.9471825063078217
# 0.9766596096400917

# Problem 14
# Parts (i - ii)

M = [
	1 101 0 0;
	2 2 102 0;
	5 0 105 0;
	2.5 2.5 2.5 102.5;
]

b = [100.8; 103.5; 107.5; 110.5]

#

# First get LU fact with pivot
fact_of_M = LU_fact(A = M, with_pivot = true)

# Then solve with LU solver
LU_linear_solver(
	P = fact_of_M["P"],
	L = fact_of_M["L"],
	U = fact_of_M["U"],
	b = b
)

# 4-element Array{Float64,1}:
# 1.0165683382497548
# 0.9879547689282202
# 0.9754015077023926
# 1.005367692319991

## Problem 15
# Part (i-ii)

x = [2/12; 5/12; 11/12; 15/12]
DFs = [.998; .9935; .982; .9775]

time_0_rate = 0.01
v = -log.(DFs)./x

# 4-element Array{Float64,1}:
# 0.012012016024038476
# 0.01565092077663711
# 0.0198152406847322
# 0.018205589698092946

#

params = create_spline_params(
	x = x,
	v = v
	time_0_rate = time_0_rate)

# Part (iii)

coupon = 2.5/4

pv_1 = coupon*exp(-1/12 * cubic_spline(t = 1/12, x = x, params = params))
pv_2 = coupon*exp(-4/12 * cubic_spline(t = 4/12, x = x, params = params))
pv_3 = coupon*exp(-7/12 * cubic_spline(t = 7/12, x = x, params = params))
pv_4 = coupon*exp(-10/12 * cubic_spline(t = 10/12, x = x, params = params))
pv_5 = (100+coupon)*exp(-13/12 * cubic_spline(t = 13/12, x = x, params = params))

pv_1+pv_2+pv_3+pv_4+pv_5 # 101.02172029298816

## Problem 16
m1 = 0.10
m2 = 0.15
m3 = 0.20

s1 = 0.15
s2 = 0.30
s3 = 0.35

r12 = -0.25
r23 = 0.20
r13 = 0.30

# Part (i)
M = [
	s1^2 r12*s1*s2 r13*s1*s3;
	r12*s1*s2 s2^2 r23*s2*s3;
	r13*s1*s3 r23*s2*s3 s3^2
]

# Part (ii)
m = [m1; m2; m3]

ons = ones(3)
zrs = zeros(2)

M_optim = [
	2 .* M ons m;
	transpose(ons) 0 0;
	transpose(m) 0 0
]

b = [0; 0; 0; 1; 0.16]

#

fact_of_M = LU_fact(A = M_optim, with_pivot = true)

fact_of_M["P"]
# 5×5 Array{Float64,2}:
# 0.0  0.0  0.0  1.0  0.0
# 0.0  1.0  0.0  0.0  0.0
# 0.0  0.0  1.0  0.0  0.0
# 1.0  0.0  0.0  0.0  0.0
# 0.0  0.0  0.0  0.0  1.0

fact_of_M["L"]
# 5×5 Array{Float64,2}:
# 1.0      0.0        0.0        0.0       0.0
# -0.0225   1.0        0.0        0.0       0.0
# 0.0315   0.0518519  1.0        0.0       0.0
# 0.045   -0.333333   0.038067   1.0       0.0
# 0.1      0.246914   0.400056  -0.482737  1.0

fact_of_M["U"]
# 5×5 Array{Float64,2}:
# 1.0  1.0     1.0       0.0        0.0      
# 0.0  0.2025  0.0645    1.0        0.15     
# 0.0  0.0     0.210156  0.948148   0.192222 
# 0.0  0.0     0.0       1.29724    0.142683 
# 0.0  0.0     0.0       0.0       -0.0450585

# Part (iii)
weights = LU_linear_solver(
	P = fact_of_M["P"],
	L = fact_of_M["L"],
	U = fact_of_M["U"],
	b = b)

# 5-element Array{Float64,1}:
#  0.23507537688442248 ## Weight in asset 1 
#  0.3298492462311555 ## Weight in asset 2
#  0.43507537688442194 ## Weight in asset 3
#  0.09412869346733665
# -1.1099035175879393 

transpose(weights[1:3]) * m # 0.15999999999999998

sqrt(transpose(weights[1:3]) * M * weights[1:3]) # 0.2042741654575213

# Part (iv)
w_1 = [.30 .20 .50]
w_2 = [.50 -0.20 .70]

exp_ret1 = w_1 * m # 0.16
exp_ret2 = w_2 * m # 0.16

w_1 = transpose(w_1)
w_2 = transpose(w_2)

exp_vol1 = sqrt(transpose(w_1) * M * w_1) # 0.2093442141545832
exp_vol2 = sqrt(transpose(w_2) * M * w_2) # 0.27684833393033087