using DataFrames
using LinearAlgebra
using Dates

cd("C:/Users/Stat-Comp-01/OneDrive/JuliaCode/MFE/LAPrimer")

### Problem 11 ###
# (Sd Diag) = sqrt(cov(i,i))
# Corr = (Sd Diag)^-1 * cov_mat * (Sd Diag)^-1

cov_mat = [
[1 -0.525 1.375 -0.075 -0.75]; 
[-0.525 2.25 0.1875 0.1875 -0.675]; 
[1.375 0.1875 6.25 0.4375 -1.875]; 
[-0.075 0.1875 0.4375 0.25 0.3]; 
[-0.75 -0.675 -1.875 0.3 9]
]

diag_inv = inv(sqrt(LinearAlgebra.Diagonal(cov_mat)))

corr_mat = diag_inv * cov_mat * diag_inv

# 5×5 Array{Float64,2}:
#   1.0   -0.35   0.55  -0.15  -0.25
#  -0.35   1.0    0.05   0.25  -0.15
#   0.55   0.05   1.0    0.35  -0.25
#  -0.15   0.25   0.35   1.0    0.2
#  -0.25  -0.15  -0.25   0.2    1.0

## Problem 12
# Cov = (Sd Diag) * Cor * (Sd Diag)

cor_mat = [
[1 -0.25 0.15 -0.05 -0.30];
[-0.25 1 -0.10 -0.25 0.10];
[0.15 -0.10 1 0.20 0.05];
[-0.05 -0.25 0.20 1 0.10];
[-0.30 0.10 0.05 0.10 1]
]

sd_1 = LinearAlgebra.Diagonal([0.25, 0.5, 1, 2, 4])
sd_2 = LinearAlgebra.Diagonal([4, 2, 1, 0.5, 0.25])

# Part (i)
cov1 = sd_1 * cor_mat * sd_1

# 5×5 Array{Float64,2}:
#   0.0625   -0.03125   0.0375  -0.025  -0.3
#  -0.03125   0.25     -0.05    -0.25    0.2
#   0.0375   -0.05      1.0      0.4     0.2
#  -0.025    -0.25      0.4      4.0     0.8
#  -0.3       0.2       0.2      0.8    16.0

# Part (ii)
cov2 = sd_2 * cor_mat * sd_2

# 5×5 Array{Float64,2}:
#  16.0  -2.0    0.6     -0.1     -0.3
#  -2.0   4.0   -0.2     -0.25     0.05
#   0.6  -0.2    1.0      0.1      0.0125
#  -0.1  -0.25   0.1      0.25     0.0125
#  -0.3   0.05   0.0125   0.0125   0.0625

### Problem 13 ###
index_prices = DataFrames.readtable("./Data/indeces-jul26-aug9-2012.csv")

#

"""
Computes percentage returns given an array of prices. Order assumed - earliest price is in the first index and latest price is in the last index.

Inputs:
	price_arr = Array of prices

Output:
	rets = Array of percentage returns in decimal format (not multiplied by 100)
"""
function perc_rets(price_arr::Array{Union{Missing, Float64},1}) where T <:Array{Union{Missing, Float64},1}

	rets = vcat(diff(price_arr),missing)./price_arr

	pop!(rets)

	return rets
end

#

# Part (i)
ret_df = DataFrame(colwise(perc_rets, index_prices[[:Dow_Jones, :NASDAQ, :S_P_500]]))

trimmed_dates = copy(index_prices[:Date])

popfirst!(trimmed_dates)

ret_df = hcat(DataFrame(Date = trimmed_dates), ret_df)

names!(ret_df, names(index_prices))

# 10×4 DataFrame
# │ Row │ Date      │ Dow_Jones    │ NASDAQ      │ S_P_500      │
# │ 1   │ 7/27/2012 │ 0.0145663    │ 0.0224108   │ 0.0190806    │
# │ 2   │ 7/30/2012 │ -0.000202667 │ -0.00414119 │ -0.000483416 │
# │ 3   │ 7/31/2012 │ -0.00492083  │ -0.0021454  │ -0.00431675  │
# │ 4   │ 8/1/2012  │ -0.00250218  │ -0.0065691  │ -0.00289998  │
# │ 5   │ 8/2/2012  │ -0.00749453  │ -0.00357509 │ -0.00750371  │
# │ 6   │ 8/3/2012  │ 0.0168718    │ 0.0199775   │ 0.0190403    │
# │ 7   │ 8/6/2012  │ 0.00162948   │ 0.00741602  │ 0.00232928   │
# │ 8   │ 8/7/2012  │ 0.00389479   │ 0.00867919  │ 0.00510676   │
# │ 9   │ 8/8/2012  │ 0.000534605  │ -0.00152859 │ 0.00062083   │
# │ 10  │ 8/9/2012  │ -0.00079313  │ 0.00245413  │ 0.00041363   │

# Part (ii)
# Tx = (Xt - mean(X)), time series matrix
# Cov = 1/(N-1) * Tx^t * Tx

ts_mat = mapslices(x -> x .- mean(x), 
	convert(Array, ret_df[[:Dow_Jones, :NASDAQ, :S_P_500]]), 
	1)

cov_mat = (1/(size(ret_df,1)-1)) * ts_mat' * ts_mat

# 3×3 Array{Float64,2}:
#  6.17416e-5  7.42765e-5   7.11061e-5
#  7.42765e-5  0.000103667  8.79881e-5
#  7.11061e-5  8.79881e-5   8.26366e-5

#

# Part (iii)
"""
Computes log returns given an array of prices. Order assumed - earliest price is in the first index and latest price is in the last index.

Inputs:
	price_arr = Array of prices

Output:
	rets = Array of log returns in decimal format (not multiplied by 100)
"""
function log_rets(price_arr::Array{Union{Missing, Float64},1}) where T <:Array{Union{Missing, Float64},1}

	rets = log.(vcat(price_arr,missing)./vcat(missing,price_arr))

	pop!(rets)
	popfirst!(rets)

	return rets
end

#

log_ret_df = DataFrame(colwise(log_rets, index_prices[[:Dow_Jones, :NASDAQ, :S_P_500]]))

trimmed_dates = copy(index_prices[:Date])

popfirst!(trimmed_dates)

log_ret_df = hcat(DataFrame(Date = trimmed_dates), log_ret_df)

names!(log_ret_df, names(index_prices))

# Part (iv)
# Tx = (Xt - mean(X)), time series matrix
# Cov = 1/(N-1) * Tx^t * Tx

ts_mat = mapslices(x -> x .- mean(x), 
	convert(Array, log_ret_df[[:Dow_Jones, :NASDAQ, :S_P_500]]), 
	1)

cov_mat = (1/(size(log_ret_df,1)-1)) * ts_mat' * ts_mat

# 3×3 Array{Float64,2}:
#  6.10747e-5  7.32119e-5   7.02163e-5
#  7.32119e-5  0.000102023  8.65872e-5
#  7.02163e-5  8.65872e-5   8.14557e-5

### Problem 14 ###
index_prices = DataFrames.readtable("./Data/indices-july2011.csv")

index_prices[:Date] = map(x -> Date(x, "m/d/y"), index_prices[:Date])

# Setting up price and return series for each time period
index_prices[:Daynum] = map(x -> day(x), index_prices[:Date])
index_prices[:Weeknum] = map(x -> week(x), index_prices[:Date])
index_prices[:Monthnum] = map(x -> month(x), index_prices[:Date])

#

daily_prices = copy(index_prices)

# Filter out max day for each week
week_inx = by(index_prices, :Weeknum, 
	df -> DataFrame(Daynum = maximum(df[:Daynum])))

weekly_prices = join(index_prices, week_inx,
	on = [:Daynum, :Weeknum])

# Filter out max day for each month
month_inx = by(index_prices, :Monthnum, 
	df -> DataFrame(Daynum = maximum(df[:Daynum])))

monthly_prices = join(index_prices, month_inx,
	on = [:Daynum, :Monthnum])

# Removing last rows from weekly and monthly series, last date was a Wednesday and didn't complete a round Month or Week
weekly_prices = weekly_prices[weekly_prices[:Date] .!= Date("2011-07-27"),:]
monthly_prices = monthly_prices[monthly_prices[:Date] .!= Date("2011-07-27"),:]

# Stripping temporal information away from price series
[delete!(df, colname) for df in [daily_prices, weekly_prices, monthly_prices], colname in [:Date,:Daynum,:Weeknum,:Monthnum]]

# Getting percentage and log returns
daily_perc_ret = convert(Array, DataFrame(colwise(perc_rets, daily_prices)))
daily_log_ret = convert(Array, DataFrame(colwise(log_rets, daily_prices)))

#

weekly_perc_ret = convert(Array, DataFrame(colwise(perc_rets, weekly_prices)))
weekly_log_ret = convert(Array, DataFrame(colwise(log_rets, weekly_prices)))

#

monthly_perc_ret = convert(Array, DataFrame(colwise(perc_rets, monthly_prices)))
monthly_log_ret = convert(Array, DataFrame(colwise(log_rets, monthly_prices)))

"""
Computes covariance and correlation matrices for a return time series.

Inputs:
	price_arr = 2D Array of prices

Output:
	Dict("cov","cor")
""" 
function cov_and_corr(ret_arr::Array{Union{Missing, Float64},2}) where T <:Array{Union{Missing, Float64},2}

	# Mean centering the data
	ts_mat = mapslices(x -> x .- mean(x), 
	ret_arr, 
	1)

	cov_mat = (1/(size(ret_arr,1)-1)) * ts_mat' * ts_mat

	# Converting to correlation matrix
	diag_inv = inv(sqrt(LinearAlgebra.Diagonal(cov_mat)))

	corr_mat = diag_inv * cov_mat * diag_inv

	# Bringing it all together
	return Dict("cov" => cov_mat, "cor" => corr_mat)
end

# Part (i)

daily_perc_mats = cov_and_corr(daily_perc_ret)

daily_perc_mats["cov"] * 1e6
# 9×9 Array{Float64,2}:
#  100.036   67.0445   96.6744  40.0444  82.7673  78.9032  74.3506   71.1056   54.5553
#   67.0445  58.1616   71.2856  38.0135  65.2921  61.3751  59.6317   52.8672   41.9855
#   96.6744  71.2856  135.876   45.3644  86.1492  83.0504  77.4632   60.1469   57.4574
#   40.0444  38.0135   45.3644  44.841   43.2031  40.1     39.365    34.5997   30.4279
#   82.7673  65.2921   86.1492  43.2031  82.3718  74.4573  71.3749   73.0498   54.0081
#   78.9032  61.3751   83.0504  40.1     74.4573  69.7742  66.9423   61.8004   46.5456
#   74.3506  59.6317   77.4632  39.365   71.3749  66.9423  65.1692   58.658    43.6568
#   71.1056  52.8672   60.1469  34.5997  73.0498  61.8004  58.658   103.964    50.0088
#   54.5553  41.9855   57.4574  30.4279  54.0081  46.5456  43.6568   50.0088  110.842

daily_perc_mats["cor"]
# 9×9 Array{Float64,2}:
#  1.0       0.878954  0.829204  0.597896  0.911783  0.944428  0.920841  0.697243  0.518091
#  0.878954  1.0       0.801884  0.744358  0.943307  0.963444  0.968586  0.679871  0.522912
#  0.829204  0.801884  1.0       0.581173  0.814311  0.852948  0.823195  0.506058  0.468189
#  0.597896  0.744358  0.581173  1.0       0.710867  0.7169    0.728202  0.506749  0.431601
#  0.911783  0.943307  0.814311  0.710867  1.0       0.982135  0.97417   0.789384  0.565219
#  0.944428  0.963444  0.852948  0.7169    0.982135  1.0       0.992732  0.725608  0.529272
#  0.920841  0.968586  0.823195  0.728202  0.97417   0.992732  1.0       0.71263   0.513663
#  0.697243  0.679871  0.506058  0.506749  0.789384  0.725608  0.71263   1.0       0.465857
#  0.518091  0.522912  0.468189  0.431601  0.565219  0.529272  0.513663  0.465857  1.0

#

daily_log_mats = cov_and_corr(daily_log_ret)

daily_log_mats["cov"] * 1e6
# 9×9 Array{Float64,2}:
#  100.4     67.2699   97.249   40.1524  83.128   79.2124  74.6269   71.3843   54.597
#   67.2699  58.3232   71.573   38.0912  65.5332  61.5761  59.8201   53.0595   41.992
#   97.249   71.573   136.667   45.4591  86.5895  83.4605  77.8154   60.39     57.5715
#   40.1524  38.0912   45.4591  44.9002  43.337   40.2007  39.4648   34.7578   30.5549
#   83.128   65.5332   86.5895  43.337   82.7236  74.761   71.6517   73.3279   54.041
#   79.2124  61.5761   83.4605  40.2007  74.761   70.0331  67.1784   62.0365   46.5754
#   74.6269  59.8201   77.8154  39.4648  71.6517  67.1784  65.3873   58.8771   43.6651
#   71.3843  53.0595   60.39    34.7578  73.3279  62.0365  58.8771  104.159    50.0584
#   54.597   41.992    57.5715  30.5549  54.041   46.5754  43.6651   50.0584  111.05

daily_log_mats["cor"]
# 9×9 Array{Float64,2}:
#  1.0       0.87909   0.830207  0.598026  0.912149  0.944658  0.921047  0.698051  0.517061
#  0.87909   1.0       0.801671  0.744354  0.943466  0.963474  0.968679  0.68076   0.521778
#  0.830207  0.801671  1.0       0.580316  0.814364  0.853095  0.823165  0.506157  0.467321
#  0.598026  0.744354  0.580316  1.0       0.711082  0.716898  0.728347  0.508253  0.43271
#  0.912149  0.943466  0.814364  0.711082  1.0       0.98222   0.974238  0.789962  0.56383
#  0.944658  0.963474  0.853095  0.716898  0.98222   1.0       0.99273   0.726353  0.528135
#  0.921047  0.968679  0.823165  0.728347  0.974238  0.99273   1.0       0.71343   0.512421
#  0.698051  0.68076   0.506157  0.508253  0.789962  0.726353  0.71343   1.0       0.465445
#  0.517061  0.521778  0.467321  0.43271   0.56383   0.528135  0.512421  0.465445  1.0

# Part (ii)
# Weekly numbers seem to be a bit different from the solution manual, unclear why

weekly_perc_mats = cov_and_corr(weekly_perc_ret)

weekly_perc_mats["cov"] * 1e6
# 9×9 Array{Float64,2}:
#  588.19   402.946  539.231  231.978  448.353  452.751  440.027  450.92   362.708
#  402.946  326.225  396.785  187.767  338.69   339.018  335.567  314.824  278.581
#  539.231  396.785  694.593  241.546  419.941  436.159  418.016  357.832  314.138
#  231.978  187.767  241.546  207.225  183.831  197.82   197.783  135.475  196.065
#  448.353  338.69   419.941  183.831  398.303  375.534  369.416  413.484  319.042
#  452.751  339.018  436.159  197.82   375.534  371.703  364.938  363.834  303.92
#  440.027  335.567  418.016  197.783  369.416  364.938  362.423  354.456  300.008
#  450.92   314.824  357.832  135.475  413.484  363.834  354.456  612.082  318.196
#  362.708  278.581  314.138  196.065  319.042  303.92   300.008  318.196  444.041

weekly_perc_mats["cor"]
# 9×9 Array{Float64,2}:
#  1.0       0.919878  0.843628  0.664457  0.926306  0.968283  0.953044  0.751512  0.709719
#  0.919878  1.0       0.833549  0.722172  0.939587  0.973569  0.975918  0.704539  0.73195
#  0.843628  0.833549  1.0       0.63667   0.798391  0.858384  0.833144  0.548793  0.565644
#  0.664457  0.722172  0.63667   1.0       0.639871  0.712774  0.721705  0.380394  0.646351
#  0.926306  0.939587  0.798391  0.639871  1.0       0.975987  0.972303  0.837428  0.758628
#  0.968283  0.973569  0.858384  0.712774  0.975987  1.0       0.994289  0.762781  0.748083
#  0.953044  0.975918  0.833144  0.721705  0.972303  0.994289  1.0       0.752574  0.747847
#  0.751512  0.704539  0.548793  0.380394  0.837428  0.762781  0.752574  1.0       0.610348
#  0.709719  0.73195   0.565644  0.646351  0.758628  0.748083  0.747847  0.610348  1.0

#

weekly_log_mats = cov_and_corr(weekly_log_ret)

weekly_log_mats["cov"] * 1e6
# 9×9 Array{Float64,2}:
#  593.995  405.557  549.034  234.763  452.803  456.855  443.293  454.405  365.082
#  405.557  327.006  401.75   188.79   340.603  340.75   336.792  315.717  278.61
#  549.034  401.75   709.64   244.256  426.826  443.059  423.745  362.415  318.805
#  234.763  188.79   244.256  209.984  185.302  199.58   199.283  136.69   196.308
#  452.803  340.603  426.826  185.302  401.387  378.448  371.729  416.51   319.975
#  456.855  340.75   443.059  199.58   378.448  374.452  367.066  365.959  305.124
#  443.293  336.792  423.745  199.283  371.729  367.066  363.968  356.237  300.489
#  454.405  315.717  362.415  136.69   416.51   365.959  356.237  615.095  320.03
#  365.082  278.61   318.805  196.308  319.975  305.124  300.489  320.03   443.34

weekly_log_mats["cor"]
# 9×9 Array{Float64,2}:
#  1.0       0.925243  0.857285  0.657426  0.932598  0.97057   0.956728  0.756774  0.716823
#  0.925243  1.0       0.849469  0.701747  0.945135  0.975953  0.97715   0.707441  0.731333
#  0.857285  0.849469  1.0       0.623177  0.816847  0.871864  0.847234  0.562923  0.579665
#  0.657426  0.701747  0.623177  1.0       0.629052  0.697877  0.709734  0.385337  0.644992
#  0.932598  0.945135  0.816847  0.629052  1.0       0.978195  0.974676  0.837016  0.75991
#  0.97057   0.975953  0.871864  0.697877  0.978195  1.0       0.994517  0.764697  0.750023
#  0.956728  0.97715   0.847234  0.709734  0.974676  0.994517  1.0       0.75714   0.750952
#  0.756774  0.707441  0.562923  0.385337  0.837016  0.764697  0.75714   1.0       0.621233
#  0.716823  0.731333  0.579665  0.644992  0.75991   0.750023  0.750952  0.621233  1.0

# Part (iii)

monthly_perc_mats = cov_and_corr(monthly_perc_ret)

monthly_perc_mats["cov"] * 1e6
# 9×9 Array{Float64,2}:
#  529.776  526.363  244.693  267.239   584.627  503.189  497.533   967.946  218.709
#  526.363  565.126  271.058  231.247   594.653  504.056  509.563   839.032  216.961
#  244.693  271.058  664.647  104.249   214.542  174.443  139.116   586.153  465.896
#  267.239  231.247  104.249  285.359   254.099  230.926  221.684   480.234  209.201
#  584.627  594.653  214.542  254.099   672.484  568.581  572.66   1076.07   162.903
#  503.189  504.056  174.443  230.926   568.581  488.239  488.07    899.597  151.062
#  497.533  509.563  139.116  221.684   572.66   488.07   494.769   852.964  119.967
#  967.946  839.032  586.153  480.234  1076.07   899.597  852.964  2609.89   397.892
#  218.709  216.961  465.896  209.201   162.903  151.062  119.967   397.892  431.368

monthly_perc_mats["cor"]
# 9×9 Array{Float64,2}:
#  1.0       0.961981  0.412363  0.68732   0.979471  0.989393  0.971793  0.823178  0.457506
#  0.961981  1.0       0.442277  0.575847  0.964606  0.9596    0.96366   0.690868  0.439425
#  0.412363  0.442277  1.0       0.239377  0.320905  0.306226  0.242594  0.445046  0.8701
#  0.68732   0.575847  0.239377  1.0       0.58005   0.618673  0.58998   0.556476  0.596272
#  0.979471  0.964606  0.320905  0.58005   1.0       0.992282  0.992783  0.812251  0.302457
#  0.989393  0.9596    0.306226  0.618673  0.992282  1.0       0.993035  0.796931  0.329165
#  0.971793  0.96366   0.242594  0.58998   0.992783  0.993035  1.0       0.750617  0.25968
#  0.823178  0.690868  0.445046  0.556476  0.812251  0.796931  0.750617  1.0       0.374999
#  0.457506  0.439425  0.8701    0.596272  0.302457  0.329165  0.25968   0.374999  1.0

#

monthly_log_mats = cov_and_corr(monthly_log_ret)

monthly_log_mats["cov"] * 1e6
# 9×9 Array{Float64,2}:
#  523.1    517.957  240.423   261.166    576.864  496.688  491.774   939.222  217.819
#  517.957  554.848  265.009   223.422    586.017  496.145  502.493   814.354  213.563
#  240.423  265.009  645.864    99.4118   211.908  172.226  137.849   578.667  460.567
#  261.166  223.422   99.4118  277.288    248.022  226.082  217.182   467.237  204.907
#  576.864  586.017  211.908   248.022    663.636  560.783  565.979  1039.55   162.041
#  496.688  496.145  172.226   226.082    560.783  481.518  482.06    871.088  150.944
#  491.774  502.493  137.849   217.182    565.979  482.06   489.535   827.201  119.558
#  939.222  814.354  578.667   467.237   1039.55   871.088  827.201  2464.77   403.04
#  217.819  213.563  460.567   204.907    162.041  150.944  119.558   403.04   434.415

monthly_log_mats["cor"]
# 9×9 Array{Float64,2}:
#  1.0       0.961422  0.413631  0.68574   0.979076  0.989658  0.97181   0.827158  0.45693
#  0.961422  1.0       0.442694  0.569604  0.965735  0.959875  0.964164  0.696367  0.434997
#  0.413631  0.442694  1.0       0.23491   0.323676  0.308832  0.245155  0.458638  0.869501
#  0.68574   0.569604  0.23491   1.0       0.578176  0.618722  0.589476  0.565177  0.59039
#  0.979076  0.965735  0.323676  0.578176  1.0       0.992028  0.992988  0.81282   0.301791
#  0.989658  0.959875  0.308832  0.618722  0.992028  1.0       0.992895  0.799592  0.330032
#  0.97181   0.964164  0.245155  0.589476  0.992988  0.992895  1.0       0.753063  0.259259
#  0.827158  0.696367  0.458638  0.565177  0.81282   0.799592  0.753063  1.0       0.3895
#  0.45693   0.434997  0.869501  0.59039   0.301791  0.330032  0.259259  0.3895    1.0

# Part (iv)
# The magnitudes of the covariance pairs typically follow this order: monthly > weekly > daily. 
# Log returns approximate percentage returns well.