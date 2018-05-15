from statsmodels.tsa.stattools import adfuller
import scipy.stats as stat
import pandas as pd
import statsmodels.api as sm

def Chow_Dickey(pd_series, split, reg_type = 'c', auto_lag = 'AIC',max_lag = 4, verbose = False):
    if isinstance(pd_series, pd.DataFrame):
        temp_index = pd_series.index
        pd_series = pd.Series(pd_series, index = temp_index)
    if isinstance(split, pd._libs.tslib.Timestamp) | isinstance(split,str):
        mask = pd_series.index<= pd.to_datetime(split)
        split1 = pd_series[mask]
        split2 =  pd_series[~mask]
    else:
        mask = pd_series.index.year<= split
        split1 = pd_series[mask]
        split2 =  pd_series[~mask]

    nr_adf = adfuller(pd_series, regression = reg_type, autolag = auto_lag, maxlag=max_lag, regresults=True)[3]
    nr_lag = nr_adf.usedlag
    nr_model = nr_adf.resols
    nr_ssr = nr_model.ssr * nr_model.nobs
    param_length = nr_model.df_model + 1


    adf1 = adfuller(split1, regression = reg_type, autolag = None, maxlag=max_lag, regresults=True)[3]
    
    adf1_model =adf1.resols
    N1 = adf1_model.nobs
    adf1_ssr = adf1_model.ssr * N1
    

    adf2 = adfuller(split2, regression = reg_type, autolag = None, maxlag=max_lag, regresults=True)[3]
    
    adf2_model =adf2.resols
    N2 = adf2_model.nobs
    adf2_ssr = adf2_model.ssr * N2

    numerator = (nr_ssr - (adf1_ssr + adf2_ssr))/param_length
    denominator = (adf1_ssr + adf2_ssr)/(N1 + N2 - 2*param_length)

    F_stat = numerator/denominator
    
    f_dist = stat.f(param_length, N1 + N2 - 2*param_length )

    p_val = 1- f_dist.cdf(F_stat)


    return F_stat , p_val,nr_lag

def Chow_test(pd_dataframe, split, reg_type = 'c', verbose = False):
    
    pd_dataframe = pd_dataframe.dropna()
    mask = pd_dataframe.index.year <= split

    split1 = pd_dataframe[mask]
    split2 = pd_dataframe[~mask]

    if reg_type == 'c':
        reg_full = sm.OLS(endog = pd_dataframe.iloc[:,0], exog = sm.add_constant(pd_dataframe.iloc[:,1:])).fit()
        reg_1 = sm.OLS(endog = split1.iloc[:,0], exog = sm.add_constant(split1.iloc[:,1:])).fit()
        reg_2 = sm.OLS(endog = split2.iloc[:,0], exog = sm.add_constant(split2.iloc[:,1:])).fit()
        

        N = reg_full.nobs
        N1 = reg_1.nobs
        N2 = reg_2.nobs
        res_full = reg_full.ssr * N
        res1 = reg_1.ssr * N1
        res2 = reg_2.ssr * N2

        param_length = reg_full.df_model + 1


    numerator = (res_full - (res1 + res2))/param_length
    denominator = (res1 + res2)/(N1 + N2 - 2*param_length)

    F_stat = numerator/denominator

    f_dist = stat.f(param_length, N1 + N2 - 2*param_length )

    p_val = 1- f_dist.cdf(F_stat)

    return F_stat , p_val, reg_full