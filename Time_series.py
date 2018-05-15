
# coding: utf-8

# In[1]:


get_ipython().magic('load_ext autoreload')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa as tsa
import statsmodels
import os
import re
import time
import sys
import datetime
import datetime
import sklearn
import statsmodels.stats.outliers_influence as outliers_influence
import gc
gc.enable()


# In[230]:


regional_ts_data = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Full_Fegional_Data.csv',parse_dates=['Date'])
regional_ts_data = regional_ts_data.set_index('Date')
real_yield = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Interest_rates/real_yield.csv',parse_dates=['Date'])
real_yield = real_yield.iloc[:,[0,7]]
real_yield.columns = ['Date','real_3yr']
real_yield= real_yield.set_index('Date')
real_yield = real_yield.resample('M').mean()

IV_matrix = regional_ts_data[['Inflation','3yr_75_ltv']].resample('M').mean()
IV_matrix=IV_matrix.join(real_yield, how= 'left')
IV_matrix.index = IV_matrix.index - pd.offsets.MonthBegin(1)
IV_full_matrix = IV_matrix.drop_duplicates().dropna()
IV_full_matrix['Real_Mortgage'] = IV_full_matrix['3yr_75_ltv']/(1+IV_full_matrix['Inflation'])
iv_reg = sm.OLS(endog = IV_full_matrix['Real_Mortgage'], exog = sm.add_constant(IV_full_matrix['real_3yr'])).fit()
iv_reg_results = iv_reg
IV_real_mortgage = iv_reg.predict(sm.add_constant(real_yield['real_3yr'])).to_frame()
IV_real_mortgage.columns = ['Real_mortgageIR_IV'] 
IV_real_mortgage['IV_Residuals'] = real_yield['real_3yr'] - IV_real_mortgage['Real_mortgageIR_IV']

IV_real_mortgage.index = IV_real_mortgage.index - pd.offsets.MonthBegin(1)
IV_real_mortgage.to_csv('C:/Users/Ben/Downloads/Project/Q1/IV/real_ir.csv')
regional_ts_data = regional_ts_data.join(IV_real_mortgage,how='left')
mapper = lambda x: 1 if  x==True else 0
vfunc = np.vectorize(mapper)
regional_ts_data['Help_to_buy'] = vfunc(regional_ts_data.index >= '2013-04-01')

print(iv_reg_results.summary())


# In[236]:


IV_tab_vals = pd.concat([iv_reg_results.params,iv_reg_results.tvalues],axis=1)
IV_tab_vals


# In[237]:


iv_BP_test = statsmodels.stats.diagnostic.het_breuschpagan(iv_reg_results.resid, sm.add_constant(IV_full_matrix['real_3yr']))[0:2]
iv_BP_test = pd.Series(iv_BP_test,name = 'Breusch-Pagan').to_frame().transpose()
iv_BP_test = pd.concat([pd.Series([[0,0]]).to_frame(),iv_BP_test],axis=0,ignore_index=True)
IV_tab_vals = pd.concat([IV_tab_vals,iv_BP_test],axis=0,ignore_index=True)
IV_tab_vals.columns = ['Coefficient','T-statistic']
IV_tab_vals.index = ['Constant','Real_3yr', '-','Breusch-Pagan LM']
# iv_BP_test


# In[249]:


iv_reg.het_scale.mean()**0.5


# In[3]:


regional_ts_data.columns


# In[4]:


region_names = regional_ts_data.Region.unique()
regional_ts_list = []
for reg_name in region_names:
    temp_ts = regional_ts_data.loc[regional_ts_data.Region == reg_name]
    regional_ts_list.append(temp_ts)

for idx, frame in enumerate(regional_ts_list):
    regional_ts_list[idx] = regional_ts_list[idx].loc[:,['Region','AveragePrice','LFS_population','dist_from_lon',
                                                         'Unemployment_rate','GDP_per_cap','Inflation_index',
                                                         'Regional_PSHS','Non_UK_born','Migrant_inflow_monthly',
                                                         'Migrant_outflow_monthly','Net_immigration_monthly',
                                                         '3yr_75_ltv','M4_supply','Yield_slope','LFS_active_perc',
                                                         'LFS_unemployed_perc','Cur_def_perc_gdp', 
                                                         'Budget_deficit', 'Quarter_indicator','Help_to_buy',
                                                         'Real_mortgageIR_IV']
                                                      ]
                                                      
    regional_ts_list[idx]['Real_GVA_per_cap'] = regional_ts_list[idx]['GDP_per_cap'] *100/ regional_ts_list[idx]['Inflation_index']
   
                                                      
    cols_to_create = ['Migrant_inflow_per_cap','Migrant_outflow_per_cap','Net_immigration_per_cap','Non_UK_per_cap','Regional_pshs_per_cap']
    cols_to_use = ['Migrant_inflow_monthly','Migrant_outflow_monthly','Net_immigration_monthly','Non_UK_born','Regional_PSHS']
    quarter_dummies = pd.get_dummies(regional_ts_list[idx].Quarter_indicator,drop_first=True)
    quarter_dummies.columns =['Q2','Q3','Q4']
    regional_ts_list[idx] = pd.concat([regional_ts_list[idx],quarter_dummies],axis=1)
    
    for new, old in zip(cols_to_create, cols_to_use):
        regional_ts_list[idx][new] = regional_ts_list[idx][old]/regional_ts_list[idx]['LFS_population']
    

matched_regions = [frame['Region'].max() for frame in regional_ts_list]
design_matrices = [frame.drop(['Region'],axis=1) for frame in regional_ts_list]
design_matrices = [dm.astype(float) for dm in design_matrices]


transform_dict = {'AveragePrice': lambda x: np.log(x),
                  'LFS_population': lambda x: np.log(x),
                  'dist_from_lon': lambda x: np.log(x),             
                  'Real_GVA_per_cap': lambda x: np.log(x),
                  'Inflation_index': lambda x: x.pct_change(),                   
                  'Regional_pshs_per_cap': lambda x: np.log(x),
                  'Non_UK_per_cap': lambda x: np.log(x),
#                   'Migrant_inflow_per_cap': lambda x: np.log(x),                                   
#                   'Migrant_outflow_per_cap': lambda x: np.log(x),
                  'Net_immigration_per_cap': lambda x: x, 
                  '3yr_75_ltv': lambda x: x,
                  'M4_supply': lambda x: np.log(x),                                 
                  'Yield_slope': lambda x: x,
                  'LFS_active_perc': lambda x: x,
                  'LFS_unemployed_perc': lambda x: x,
                  'Cur_def_perc_gdp': lambda x: x,
                  'Budget_deficit': lambda x: x, 
                  'Q2':lambda x: x,
                  'Q3':lambda x: x,
                  'Q4': lambda x: x,
                 'Help_to_buy': lambda x: x,
                 'Real_mortgageIR_IV': lambda x: x}

trans_design_matrices = list(design_matrices)
for idx, dm in enumerate(trans_design_matrices):
    trans_design_matrices[idx] = trans_design_matrices[idx].transform(transform_dict)
    #Mortgage IV is null from 2016 on
#     trans_design_matrices[idx] = trans_design_matrices[idx].dropna()


# In[5]:


#############################


# In[6]:


# Constant only test

# max_lags = 15

# struct_break_list = list(trans_design_matrices)
# mapper = lambda x: 1 if  x==True else 0
# vfunc = np.vectorize(mapper)
# for idx, dm in enumerate(struct_break_list):
#     struct_break_list[idx] = struct_break_list[idx].loc[:,['AveragePrice']]
#     struct_break_list[idx]['y_lag_1'] = struct_break_list[idx]['AveragePrice'].shift(1)
#     struct_break_list[idx]['y_diff'] = struct_break_list[idx]['AveragePrice'] - struct_break_list[idx]['y_lag_1']
#     for i in np.arange(1,max_lags +1):
#         struct_break_list[idx]['y_diff_lag_'+str(i)] = struct_break_list[idx]['y_diff'].shift(i)
        
#     for date in struct_break_list[idx].index.year.unique():
#         struct_break_list[idx]['on_break_dummy_'+str(str(date)[:10])] = vfunc(struct_break_list[idx].index.year == date)
#         struct_break_list[idx]['post_break_dummy_'+str(str(date)[:10])] = vfunc(struct_break_list[idx].index.year >= date)

# BICS = {}
# Results = {}
# for (idx1, dm), region_name in zip(enumerate(struct_break_list),region_names):
#     for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):         
#         for idx3, lag_length in enumerate(np.arange(1,max_lags+1)):  
#             exog_indices = [1] + [3 + j for j in range(lag_length)] + [max_lags +3 +idx2*2 , max_lags + 4 +idx2*2]
#             design_matrix = struct_break_list[idx1].iloc[:,[2] + exog_indices].dropna()
#             endog =  design_matrix.iloc[:,0]
#             exog = sm.add_constant(design_matrix.iloc[:,1:])
#             if True:#date>=datetime.datetime.strptime('1996-01-01', '%Y-%m-%d'):
#                 res = sm.OLS(endog = endog, exog = exog).fit()
#                 BICS[region_name+ '_' + str(date)[:10] + '_' + str(lag_length)] = res.bic
#                 Results[region_name+ '_' + str(date)[:10] + '_' + str(lag_length)] = res


# optimal_lags_dict = {}
# for (idx1, dm), region_name in zip(enumerate(struct_break_list),region_names):
#     for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):
#         temp_list = []
#         for idx3, lag_length in enumerate(np.arange(1,max_lags + 1)):
#             temp_val = BICS[region_name+ '_' + str(date)[:10] + '_' + str(lag_length)]
#             temp_list.append(temp_val)
        
#         opt_lag_length = np.argmax(temp_list) + 1
#         optimal_lags_dict[region_name+ '_' + str(date)[:10]] = opt_lag_length


# pot_struct_breaks = []
# for idx1, region_name in enumerate(region_names):
#     for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):
#         if Results[region_name+ '_' + str(date)[:10] + '_' + str(4)].tvalues['y_lag_1'] < -2.872:
#             pot_struct_breaks.append(region_name+ '_' + str(date)[:10] + '_' + str(4))
            


# In[92]:


#constant and trend tests for first differences (I(2))

max_lags = 15

struct_break_list = list(trans_design_matrices)
mapper = lambda x: 1 if  x==True else 0
vfunc = np.vectorize(mapper)

for idx, dm in enumerate(struct_break_list):
    struct_break_list[idx] = struct_break_list[idx].loc[:,['AveragePrice']].diff()
    struct_break_list[idx]['y_lag_1'] = struct_break_list[idx]['AveragePrice'].shift(1)
    struct_break_list[idx]['y_diff'] = struct_break_list[idx]['AveragePrice'] - struct_break_list[idx]['y_lag_1']
    struct_break_list[idx]['trend'] = np.arange(len(struct_break_list[idx]))
    for i in np.arange(1,max_lags +1):
        struct_break_list[idx]['y_diff_lag_'+str(i)] = struct_break_list[idx]['y_diff'].shift(i)
        
    for date in struct_break_list[idx].index.year.unique():
        struct_break_list[idx]['on_break_dummy_'+str(str(date)[:10])] = vfunc(struct_break_list[idx].index.year == date)
        struct_break_list[idx]['post_break_dummy_'+str(str(date)[:10])] = vfunc(struct_break_list[idx].index.year >= date)
        mask = struct_break_list[idx].index.year == date
        struct_break_list[idx]['post_break_trend_'+str(str(date)[:10])] = struct_break_list[idx]['trend'] * vfunc(mask)
        
BICS = {}
Results = {}
for (idx1, dm), region_name in zip(enumerate(struct_break_list),region_names):
    for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):         
        for idx3, lag_length in enumerate(np.arange(1,max_lags+1)):  
            exog_indices = [1,3] + [4 + j for j in range(lag_length)] + [max_lags +4 +idx2*3 , max_lags + 5 +idx2*3, max_lags + 6 +idx2*3 ]
            design_matrix = struct_break_list[idx1].iloc[:,[2] + exog_indices].dropna()
            endog =  design_matrix.iloc[:,0]
            exog = sm.add_constant(design_matrix.iloc[:,1:])
            if True:#date>=datetime.datetime.strptime('1996-01-01', '%Y-%m-%d'):
                res = sm.OLS(endog = endog, exog = exog).fit()
                BICS[region_name+ '_' + str(date)[:10] + '_' + str(lag_length)] = res.bic
                Results[region_name+ '_' + str(date)[:10] + '_' + str(lag_length)] = res


optimal_lags_dict = {}
for (idx1, dm), region_name in zip(enumerate(struct_break_list),region_names):
    for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):
        temp_list = []
        for idx3, lag_length in enumerate(np.arange(1,max_lags + 1)):
            temp_val = BICS[region_name+ '_' + str(date)[:10] + '_' + str(lag_length)]
            temp_list.append(temp_val)
        
        opt_lag_length = np.argmax(temp_list) + 1
        optimal_lags_dict[region_name+ '_' + str(date)[:10]] = opt_lag_length


pot_struct_breaks_m = []
for idx1, region_name in enumerate(region_names):
    for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):
        if Results[region_name+ '_' + str(date)[:10] + '_' + str(4)].tvalues['y_lag_1'] < -3.4264:
            pot_struct_breaks_m.append(region_name+ '_' + str(date)[:10] + '_' + str(4))
            
                        
pot_struct_breaks_norm = []
for idx1, region_name in enumerate(region_names):
    for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):
        if Results[region_name+ '_' + str(date)[:10] + '_' + str(4)].tvalues['y_lag_1'] < -3.43:
            pot_struct_breaks_norm.append(region_name+ '_' + str(date)[:10] + '_' + str(4))
        
non_stationary_series_m = []
for idx1, region_name in enumerate(region_names):
    for idx2, date in enumerate(struct_break_list[idx].index.year.unique()):
        if Results[region_name+ '_' + str(date)[:10] + '_' + str(4)].tvalues['y_lag_1'] > -3.4264:
            pot_struct_breaks_m.append(region_name+ '_' + str(date)[:10] + '_' + str(4))
                


# In[93]:


pot_struct_breaks_m


# In[64]:


beg_kat = {}
for reg_name, year in zip(break_df.index, break_df['Sup-LM break Year']):
    temp_res = Results[reg_name+ '_' + str(date)[:10] + '_' + str(4)]
    beg_kat[reg_name] = temp_res.tvalues['y_lag_1']
beg_kat = pd.DataFrame.from_dict(beg_kat,orient = 'index')
beg_kat.columns=['Robust ADF statistic']
table2 = break_df.join(beg_kat)
table2["Mackinnon's P-value"] = -3.4267


# In[9]:


Results[pot_struct_breaks_m[4]].summary()


# In[10]:


################################################################


# In[85]:


get_ipython().magic('autoreload')
import sys
sys.path.append('C:/Users/Ben/Downloads/Project/Q1/Metrics_IIA_Project_code/')
from Chow_dickey_test import *
import heapq

break_dict = {}

for idx1, region_name in enumerate(region_names):
    F_results = []
    P_vals = []
    for idx2, date in enumerate(trans_design_matrices[idx1].index.year.unique()):
        try:
            f_stat,p_val,lag = Chow_Dickey(pd_series = trans_design_matrices[idx1]['AveragePrice'], split=date ,reg_type = 'ct',max_lag=3)
            print(region_name,lag)
        except ValueError:
            f_stat = 0
            p_val = 1.1
        F_results.append(f_stat)
        P_vals.append(p_val)
        
    break_dict[region_name] = [trans_design_matrices[idx1].index.year.unique()[heapq.nlargest(1, range(len(F_results)), key = F_results.__getitem__)][0], max(F_results)]
    


# In[86]:


break_dict


# In[20]:


break_df = pd.DataFrame.from_dict(break_dict, orient = 'index')
break_df.columns= ['Sup-LM break Year', 'Sup-LM statistic']
break_df['Sup-LM Critical Value'] = 3.15
break_df.loc['Northern Ireland','Sup-LM Critical Value'] = 4.71
break_df


# In[ ]:


#############################


# In[21]:


unit_root_results = {}
for (idx, dm), region_name in zip(enumerate(trans_design_matrices),region_names):
    design_matrix = dm.loc[:,'AveragePrice']
    adf = tsa.stattools.adfuller(design_matrix,maxlag=4,autolag='BIC', regression='ct')
    unit_root_results[region_name] = adf
    print(region_name + ':',adf[1])


# In[22]:


print(region_names)


# In[23]:


tsa.stattools.adfuller(trans_design_matrices[0][:'2012']['AveragePrice'],maxlag=4, regression='ct')[1]
tsa.stattools.adfuller(trans_design_matrices[0]['2012':]['AveragePrice'],maxlag=4, regression='ct')[1]


# In[24]:


tsa.stattools.adfuller(trans_design_matrices[0]['2012':]['AveragePrice'],maxlag=4, regression='ct',regresults=True)[3].resols.nobs


# In[25]:


test = tsa.stattools.adfuller(trans_design_matrices[0]['2012':]['AveragePrice'],maxlag=4, regression='ct',regresults=True)[3]
test.resols.ssr


# In[26]:


tsa.stattools.adfuller(trans_design_matrices[0][:'2012']['AveragePrice'],autolag=False, regression='ct')[1]
tsa.stattools.adfuller(trans_design_matrices[0]['2012':]['AveragePrice'],maxlag=4, regression='ct')[1]


# In[27]:


tsa.stattools.adfuller(trans_design_matrices[5]['2007':]['AveragePrice'],maxlag=4, regression='ct')[1]
tsa.stattools.adfuller(trans_design_matrices[5][:'2001']['AveragePrice'],maxlag=4, regression='ct')[1]



# In[28]:


tsa.stattools.adfuller(trans_design_matrices[7]['2007':]['AveragePrice'],maxlag=4, regression='ct')[1]
tsa.stattools.adfuller(trans_design_matrices[7][:'2007']['AveragePrice'],maxlag=4, regression='ct')[1]


# In[29]:


tsa.stattools.adfuller(trans_design_matrices[9][:'2003']['AveragePrice'],maxlag=4, regression='ct')[1]
tsa.stattools.adfuller(trans_design_matrices[8]['2003':]['AveragePrice'],maxlag=4, regression='ct')[1]


# In[30]:


##############################################################################


# In[31]:


type(regional_ts_data.index[0])


# In[268]:


cols_to_test = ['LFS_population', 'Real_GVA_per_cap','Regional_pshs_per_cap',
                 'Non_UK_per_cap', 'Net_immigration_per_cap', 'M4_supply', 'Yield_slope',
                 'LFS_active_perc','LFS_unemployed_perc', 'Cur_def_perc_gdp', 'Budget_deficit',
                 'Real_mortgageIR_IV']

stationarity_dictT = {}
stationarity_dictP = {}
usable_regions_indices = [0,1,2,3,4,7,8,9,10,11]
for regidx in usable_regions_indices:
    for colname in cols_to_test:
        adf = tsa.stattools.adfuller(trans_design_matrices[regidx][:'2007'][colname].dropna(),maxlag=4,autolag='BIC', regression='ct')
        stationarity_dictT[region_names[regidx]+ colname] = adf[0]
        stationarity_dictP[region_names[regidx]+ colname] = adf[1]

dickeyF_pvals  = pd.DataFrame(pd.DataFrame.from_dict(stationarity_dictP,orient='index').values.reshape((len(usable_regions_indices),-1)),index = region_names[usable_regions_indices], columns = cols_to_test)
dickeyF_tvals = pd.DataFrame(pd.DataFrame.from_dict(stationarity_dictT,orient='index').values.reshape((len(usable_regions_indices),-1)),index = region_names[usable_regions_indices], columns =cols_to_test)



stationarityI1 = pd.concat([dickeyF_tvals, dickeyF_pvals]).sort_index(kind='mergesort')


# In[411]:


stationarity_dictT2 = {}
stationarity_dictP2 = {}
usable_regions_indices = [0,1,2,3,4,7,8,9,10,11]
for regidx,reg_name in enumerate(region_names):
    for colname in cols_to_test:
        try:
            adf = tsa.stattools.adfuller(trans_design_matrices[regidx].diff()['2007':][colname].dropna(),maxlag=4,autolag='BIC', regression='ct')
            stationarity_dictT2[region_names[regidx]+ colname] = adf[0]
            stationarity_dictP2[region_names[regidx]+ colname] = adf[1]
        except:
            stationarity_dictT2[region_names[regidx]+ colname] = np.nan
            stationarity_dictP2[region_names[regidx]+ colname] = np.nan

dickeyF_pvals2  = pd.DataFrame(pd.DataFrame.from_dict(stationarity_dictP2,orient='index').values.reshape((len(region_names),-1)),index = region_names, columns = cols_to_test)
dickeyF_tvals2 = pd.DataFrame(pd.DataFrame.from_dict(stationarity_dictT2,orient='index').values.reshape((len(region_names),-1)),index = region_names, columns =cols_to_test)

stationarityI2 = pd.concat([dickeyF_tvals2, dickeyF_pvals2]).sort_index(kind='mergesort')


# In[412]:


stationarityI2


# In[402]:


i2_cols = dickeyF_pvals2.columns[(dickeyF_pvals2 >0.05).all()].tolist()
i2_cols


# In[49]:


##################################


# In[50]:


from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
# plot_acf((trans_design_matrices[0]['AveragePrice'] - trans_design_matrices[0]['AveragePrice'].shift(1)).dropna(), alpha =0.05)
trans_design_matrices[2][['Real_mortgageIR_IV']].plot()
plt.show()


# In[51]:


tsa.stattools.adfuller(trans_design_matrices[4]['AveragePrice'],autolag='BIC', regression='ct')[1]


# In[52]:


######################################


# In[53]:


ripple_matrices = trans_design_matrices.copy()

for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            ripple_matrices[idx1][reg_name2 + '_HP'] = ripple_matrices[idx2]['AveragePrice']


# In[89]:


from statsmodels.tsa.stattools import grangercausalitytests
for region in region_names:
    region_to_test = region
    HPcolumn_name = region_to_test +'_HP'

    granger_tests = []
    for idx, reg_name in enumerate(region_names):
        if reg_name ==region_to_test:
            granger_tests.append(None)
        else:
            test = grangercausalitytests(ripple_matrices[idx]['2007':][['AveragePrice',HPcolumn_name]],maxlag = 24, verbose = False)
            granger_tests.append(test)
            
    for idx2, test in enumerate(granger_tests):
        if not test == None:
            if test[24][0]['ssr_ftest'][1] <0.01:
                print(region, 'causing', region_names[idx2] +': \n')
                print('SSR_test:',test[24][0]['ssr_ftest'], 'LR_test:', test[24][0]['lrtest']) 


# In[174]:


from statsmodels.tsa.stattools import grangercausalitytests

granger_dict1 = {}
granger_dict2 = {}
for region in region_names:
    region_to_test = region
    HPcolumn_name = region_to_test +'_HP'

    granger_tests = []
    for idx, reg_name in enumerate(region_names):
        if reg_name ==region_to_test:
            granger_tests.append(None)
        else:
            test = grangercausalitytests(ripple_matrices[idx]['2007':][['AveragePrice',HPcolumn_name]].diff().dropna(),maxlag = 24, verbose = False)
            granger_tests.append(test)
   
    granger_dict1[region] = [None if test == None else test[24][0]['lrtest'][0] for test in granger_tests]
    granger_dict2[region] = [None if test == None else test[24][0]['lrtest'][1] for test in granger_tests]
    for idx2, test in enumerate(granger_tests):
        if not test == None:
            if test[24][0]['ssr_ftest'][1] <0.01:
                print(region, 'causing', region_names[idx2] +': \n')
                print('SSR_test:',test[24][0]['ssr_ftest'], 'LR_test:', test[24][0]['lrtest']) 


# In[179]:


GC_df1 = pd.DataFrame.from_dict(granger_dict1)
GC_df2 = pd.DataFrame.from_dict(granger_dict2)
GC_df = pd.concat([GC_df1, GC_df2]).sort_index(kind= 'mergesort')
GC_df.index = [region_names[int(i/2)] if i%2==0 else i for i in range(len(GC_df))]
GC_df


# In[110]:


for test in granger_tests:
    print('SSR_test:',test[24][0]['ssr_ftest'], 'LR_test:', test[24][0]['lrtest'])


# In[ ]:


#Whole time period
from statsmodels.tsa.stattools import coint

for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            trans_design_matrices[idx1][reg_name2 + '_HP'] = trans_design_matrices[idx2]['AveragePrice']

integrated_cols = dickeyF_pvals.columns[(dickeyF_pvals >0.05).all()].tolist()


integrated_cols_list = [integrated_cols for i in range(len(region_names))]
for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            integrated_cols_list[idx1] = integrated_cols_list[idx1] + [reg_name2 + '_HP']


coint_results = {}           
for idx, reg_name in enumerate(region_names):
    temp = ripple_matrices[idx].loc[:,integrated_cols_list[idx]].dropna()
    
    
    temp_reg = sm.OLS(temp.iloc[:,0], sm.add_constant(temp.iloc[:,1:])).fit()
    
    while ((temp_reg.pvalues > 0.05).any()) &(len(temp_reg.pvalues)>= 3):
        temp = temp.drop(temp_reg.pvalues.argmin(), axis =1)
        temp_reg = sm.OLS(temp.iloc[:,0], temp.iloc[:,1:]).fit()
    
    resid = pd.Series(data = temp_reg.resid, index = temp.index,name = 'resid')
    temp_adf = adfuller(temp_reg.resid)[0]
    
    
    coint_results[reg_name] = [temp_reg,temp_adf, resid]
        


ECMS= []
for idx, reg_name in enumerate(region_names):
    mod_resid = coint_results[reg_name][2]

    temp_dm = ripple_matrices[idx][integrated_cols].diff().join(mod_resid)

    temp_dm = temp_dm.dropna()
    for i in [1]:
        temp_dm['lag_resid_'+ str(i)] = temp_dm['resid'].shift(i)
    temp_dm = temp_dm.drop('resid',axis =1)
    temp_dm = temp_dm.dropna()
    reg = sm.OLS(endog = temp_dm.iloc[:,0], exog = sm.add_constant(temp_dm.iloc[:,1:])).fit()
    ECMS.append(reg)
    

for i in range(len(ECMS)):
    print(ECMS[i].pvalues[ECMS[i].pvalues<0.05])


# In[ ]:


#Up to end of 2006 results for ECM + coint
from statsmodels.tsa.stattools import coint

for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            trans_design_matrices[idx1][reg_name2 + '_HP'] = trans_design_matrices[idx2]['AveragePrice']

integrated_cols = dickeyF_pvals.columns[(dickeyF_pvals >0.05).all()].tolist()


integrated_cols_list = [integrated_cols for i in range(len(region_names))]
for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            integrated_cols_list[idx1] = integrated_cols_list[idx1] + [reg_name2 + '_HP']


coint_results = {}           
for idx, reg_name in enumerate(region_names):
    temp = ripple_matrices[idx][:'2006'].loc[:,integrated_cols_list[idx]].dropna()
    
    
    temp_reg = sm.OLS(temp.iloc[:,0], sm.add_constant(temp.iloc[:,1:])).fit()
    
    while ((temp_reg.pvalues > 0.05).any()) &(len(temp_reg.pvalues)>= 3):
        temp = temp.drop(temp_reg.pvalues.argmin(), axis =1)
        temp_reg = sm.OLS(temp.iloc[:,0], temp.iloc[:,1:]).fit()
    
    resid = pd.Series(data = temp_reg.resid, index = temp.index,name = 'resid')
    temp_adf = adfuller(temp_reg.resid)[0]
    
    
    coint_results[reg_name] = [temp_reg,temp_adf, resid]
        


ECMS= []
for idx, reg_name in enumerate(region_names):
    mod_resid = coint_results[reg_name][2]

    temp_dm = ripple_matrices[idx][:'2006'][integrated_cols].diff().join(mod_resid)

    temp_dm = temp_dm.dropna()
    for i in range(24):
        temp_dm['lag_resid_'+ str(i+1)] = temp_dm['resid'].shift(i+1)
    temp_dm = temp_dm.drop('resid',axis =1)
    temp_dm = temp_dm.dropna()
    reg = sm.OLS(endog = temp_dm.iloc[:,0], exog = sm.add_constant(temp_dm.iloc[:,1:])).fit()
    ECMS.append(reg)
    

for i in range(len(ECMS)):
    print(ECMS[i].pvalues[ECMS[i].pvalues<0.05])


# In[ ]:


#>= 2007 results for ECM + coint
from statsmodels.tsa.stattools import coint

for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            trans_design_matrices[idx1][reg_name2 + '_HP'] = trans_design_matrices[idx2]['AveragePrice']

integrated_cols = dickeyF_pvals.columns[(dickeyF_pvals >0.05).all()].tolist()


integrated_cols_list = [integrated_cols for i in range(len(region_names))]
for idx1, reg_name1 in enumerate(region_names):
    for idx2, reg_name2 in enumerate(region_names):
        if reg_name1 != reg_name2:
            integrated_cols_list[idx1] = integrated_cols_list[idx1] + [reg_name2 + '_HP']


coint_results = {}           
for idx, reg_name in enumerate(region_names):
    temp = ripple_matrices[idx]['2007':].loc[:,integrated_cols_list[idx]].dropna()
    
    
    temp_reg = sm.OLS(temp.iloc[:,0], sm.add_constant(temp.iloc[:,1:])).fit()
    
    while ((temp_reg.pvalues > 0.05).any()) &(len(temp_reg.pvalues)>= 3):
        temp = temp.drop(temp_reg.pvalues.argmin(), axis =1)
        temp_reg = sm.OLS(temp.iloc[:,0], temp.iloc[:,1:]).fit()
    
    resid = pd.Series(data = temp_reg.resid, index = temp.index,name = 'resid')
    temp_adf = adfuller(temp_reg.resid)[0]
    
    
    coint_results[reg_name] = [temp_reg,temp_adf, resid]
  


# In[340]:


Drake_coint_results_regc = {}
Drake_coint_results_regt = {}
Drake_coint_results_adf = {}
Drake_coint_results =[]
coint_cols = ['AveragePrice','Real_GVA_per_cap','Regional_pshs_per_cap', 'Real_mortgageIR_IV']

for idx, region_name in enumerate(region_names):
    temp = trans_design_matrices[idx]['2007':].loc[:,coint_cols].dropna()
    temp_reg = sm.OLS(temp.iloc[:,0], sm.add_constant(temp.iloc[:,1:])).fit()
    resid = pd.Series(data = temp_reg.resid, index = temp.index,name = 'resid')
    temp_adf = adfuller(temp_reg.resid)[1]
    
    Drake_coint_results.append(temp_reg)
    Drake_coint_results_regc[region_name] = temp_reg.params
    Drake_coint_results_regt[region_name] = temp_reg.tvalues
    Drake_coint_results_adf[region_name] = temp_adf
    


# In[341]:


Drake_param = pd.DataFrame.from_dict(Drake_coint_results_regc)
Drake_t = pd.DataFrame.from_dict(Drake_coint_results_regt)
Drake_adf = pd.DataFrame.from_dict(Drake_coint_results_adf, orient='index')


# In[342]:


Drake_table = pd.concat([Drake_param, Drake_t]).sort_index(kind='mergesort')


# In[356]:


get_ipython().magic('autoreload')
from Latex_table_maker import *

stat1 = np.array([statsmodels.stats.diagnostic.het_breuschpagan(stat.resid,sm.add_constant(trans_design_matrices[idx]['2007':].loc[:,coint_cols].dropna().iloc[:,1:]))[1] for idx,stat in enumerate(Drake_coint_results)])
temp_reg.ssr.mean()
print(Add_Individual_Row(stat1,''))


# In[352]:


test = statsmodels.stats.outliers_influence.reset_ramsey(temp_reg,degree=2)
test.fvalue


# In[105]:


for reg_name in region_names:
    print(reg_name, '\n',coint_results[reg_name][0].pvalues)


# In[282]:


coint_results = {}           
for idx, reg_name in enumerate(region_names):
    temp = ripple_matrices[idx]['2007':].loc[:,['AveragePrice','Real_GVA_per_cap','Regional_pshs_per_cap', 'Real_mortgageIR_IV']].dropna()
    
    
    temp_reg = sm.OLS(temp.iloc[:,0], sm.add_constant(temp.iloc[:,1:])).fit()
    
    while ((temp_reg.pvalues > 0.01).any()) &(len(temp_reg.pvalues)>= 3):
        temp_reg = sm.OLS(temp.iloc[:,0], temp.iloc[:,1:]).fit()
        temp = temp.drop(temp_reg.pvalues.argmin(), axis =1)
        
    resid = pd.Series(data = temp_reg.resid, index = temp.index,name = 'resid')
    temp_adf = adfuller(temp_reg.resid)[0]
    
    
    coint_results[reg_name] = [temp_reg,temp_adf, resid]


# In[122]:


for reg_name in region_names:
    print(reg_name, '\n',coint_results[reg_name][0].pvalues)


# In[359]:



ECMS= []
for idx, reg_name in enumerate(region_names):
    mod_resid = Drake_coint_results[idx].resid
    mod_resid.name = 'resid'

    temp_dm = trans_design_matrices[idx]['2007':][integrated_cols].diff().join(mod_resid)

    temp_dm = temp_dm.dropna()
#     for i in range(24):
#         temp_dm['lag_resid_'+ str(i+1)] = temp_dm['resid'].shift(i+1)
    
    temp_dm['lag_resid_'+ str(1)] = temp_dm['resid'].shift(1)
    temp_dm = temp_dm.drop('resid',axis =1)
    temp_dm = temp_dm.dropna()
    reg = sm.OLS(endog = temp_dm.iloc[:,0], exog = temp_dm.iloc[:,1:]).fit(cov_type='HC0')
    ECMS.append(reg)
    

for i in range(len(ECMS)):
    print(region_names[i],'\n', pd.concat([ECMS[i].params[ECMS[i].pvalues<0.05],ECMS[i].pvalues[ECMS[i].pvalues<0.05]],axis=1),'\n')


# In[382]:


get_ipython().magic('autoreload')
ECM_results = {}
ECM_results1 = []
for idx, reg_name in enumerate(region_names):
    temp = pd.concat([ECMS[idx].params, ECMS[idx].tvalues],axis =0).sort_index(kind='mergesort')
    ECM_results[reg_name] = temp
    ECM_results1.append(temp)


# test0 = Latex_table_from_pandas(table1 = pd.DataFrame.from_dict(ECM_results),table_label=5,caption='Error Correction Model results',Column_Var = 'Region', Row_Var='Exogenous Variable')


# In[400]:


get_ipython().magic('autoreload')
stat_list = []
from Latex_table_maker import *
for idx, reg_name in enumerate(region_names):
    mod_resid = Drake_coint_results[idx].resid
    mod_resid.name = 'resid'

    temp_dm = trans_design_matrices[idx]['2007':][integrated_cols].diff().join(mod_resid)

    temp_dm = temp_dm.dropna()

    
    temp_dm['lag_resid_'+ str(1)] = temp_dm['resid'].shift(1)
    temp_dm = temp_dm.drop('resid',axis =1)
    temp_dm = temp_dm.dropna()
    reg = sm.OLS(endog = temp_dm.iloc[:,0], exog = temp_dm.iloc[:,1:]).fit(cov_type='HC0')

    stat_list.append((reg.rsquared_adj))
print(Add_Individual_Row(stat_list,''))


# In[365]:


for i in range(len(ECMS)):
    print(region_names[i],'\n',statsmodels.stats.outliers_influence.reset_ramsey(ECMS[i],degree =3))


# In[141]:



for idx, reg_name in enumerate(region_names):
    mod_resid = coint_results[reg_name][2]
    temp_dm = ripple_matrices[idx]['2007':][integrated_cols].diff().join(mod_resid)

    temp_dm = temp_dm.dropna()
    temp_dm['lag_resid_'+ str(1)] = temp_dm['resid'].shift(1)
    temp_dm = temp_dm.drop('resid',axis =1)
    temp_dm = temp_dm.dropna()
    exog = temp_dm.iloc[:,1:]
    Br_Pag = statsmodels.stats.diagnostic.het_breuschpagan(ECMS[i].resid, exog_het = exog)
    print(Br_Pag)


# In[ ]:


test = statsmodels.tsa.vector_ar.var_model.VAR(endog = trans_design_matrices[0]['2007':][['AveragePrice','Real_GVA_per_cap', 'Regional_pshs_per_cap','Real_mortgageIR_IV']].diff().dropna(), freq=None, missing='none').fit(maxlags = 6)


# In[ ]:


test.summary()


# In[ ]:


test.pvalues


# In[ ]:


design_matrices[0].iterrows()


# In[273]:


get_ipython().magic('autoreload')
sys.path.append('C:/Users/Ben/Downloads/Project/Q1/Metrics_IIA_Project_code/')
from Latex_table_maker import *

ltable2= Latex_table_from_pandas(table1= stationarityI1,table2=stationarityI2, table_label=4,caption='I(1)/I(2) ADF tests and Cointegration Results',Column_Var = 'Exogenous Variable', Row_Var='Region')


# In[310]:


def Add_Individual_Row(Row,  Latex_String,sf=3,num_table_cols = len(Drake_adf.values)+1, row_name = 'insert name'):
       Latex_String1 = Latex_String + '{\\bfseries '+ row_name +'}&'
       for idx, j in enumerate(Row):
           if idx == len(Row)-1:
               if isinstance(j,float):
                   Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
               elif isinstance(j,int):
                   Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
               else:
                   try:
                       Latex_String1 += '' + str(round(float(j),sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
                   except:
                       Latex_String1 += '' + str(j) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'

           else:
               if isinstance(j,float):
                   Latex_String1 += ''+str(round(j,sf))+ '&'
               elif isinstance(j,int):
                   Latex_String1 += ''+str(round(j,sf))+ '&' 
               else:
                   try:
                       Latex_String1 += ''+str(round(float(j),sf))+ '&'
                   except:
                       Latex_String1 += ''+str(j)+ '&'
       return Latex_String1
   
test = Add_Individual_Row(Drake_adf.values,'',sf=3)
print(test)


# In[309]:


get_ipython().magic('autoreload')
sys.path.append('C:/Users/Ben/Downloads/Project/Q1/Metrics_IIA_Project_code/')
from Latex_table_maker import *

ltable2= Specific_Latex_table_from_pandas(table1= Drake_table, table_label=4,caption='I(1)/I(2) ADF tests and Cointegration Results',Column_Var = 'Region', Row_Var='Regressor')

