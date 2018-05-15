
# coding: utf-8

# In[1]:


get_ipython().magic('load_ext autoreload')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa as tssm
import statsmodels
import sys
import os
import re
import time
import datetime
import sklearn
import statsmodels.stats.outliers_influence as outliers_influence
import gc
gc.enable()


# In[2]:


cross_section = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Cross_Section_Data.csv')
cross_section = cross_section[cross_section.Region.notnull()] 


# In[3]:


design_matrix = cross_section.loc[:,['Region','RegionName','AveragePrice','Population','dist_from_lon',
                                     'Unemployment_rate','GDP_per_cap','Inflation_index','Inflation',
                                     'Local_pshs','Outstanding_perc','Good_perc',
                                     'Inad_perc','Non_UK_born','Migrant_inflow_monthly',
                                     'Migrant_outflow_monthly','Net_immigration_monthly',
                                     'LFS_active_perc']]
design_matrix['Real_GVA_per_cap'] = design_matrix['GDP_per_cap'] *100/ design_matrix['Inflation_index']

from sklearn.preprocessing import Imputer

design_matrix[['Outstanding_perc','Good_perc']] =Imputer(strategy = 'median').fit_transform(design_matrix[['Outstanding_perc','Good_perc']])
design_matrix[['Inad_perc']] =Imputer(strategy = 'mean').fit_transform(design_matrix[['Inad_perc']])

design_matrix[['Local_pshs']] = Imputer(strategy = 'median').fit_transform(design_matrix[['Local_pshs']])

cols_to_create = ['Migrant_inflow_per_cap','Migrant_outflow_per_cap','Net_immig_per_cap','Non_UK_per_cap','Local_pshs_per_cap']
cols_to_use = ['Migrant_inflow_monthly','Migrant_outflow_monthly','Net_immigration_monthly','Non_UK_born','Local_pshs']
    
for new, old in zip(cols_to_create, cols_to_use):
    design_matrix[new] = design_matrix[old]/design_matrix['Population']



region_dummies = pd.get_dummies(cross_section.loc[:,['Region']],drop_first = True)
dummy_names = region_dummies.columns.values

design_matrix = design_matrix.drop(['Region','RegionName','GDP_per_cap','Inflation_index',
                                    'Migrant_inflow_monthly','Migrant_outflow_monthly','Net_immigration_monthly',
                                    'Non_UK_born','Inflation'],axis=1)
design_matrix = design_matrix.astype(float)

transform_dict0 = {'AveragePrice': np.log,
                   'dist_from_lon':lambda x: np.log(x +1),
                   'Real_GVA_per_cap':np.log,
                  'Unemployment_rate': lambda x: x,                                                      
#                   'Local_pshs_per_cap':lambda x: np.log(x+1),
                   'Local_pshs':lambda x: np.log(x+1),
                  'Outstanding_perc':lambda x: np.log(x+1),
                  'Good_perc': lambda x: np.log(x+1),
                  'Inad_perc' : lambda x: np.log(x+1),
                  'Outstanding_perc':lambda x: x,
                  'Good_perc': lambda x: x,
                  'Inad_perc' : lambda x: x,
#                   'Non_UK_per_cap':np.log,
#                   'Migrant_inflow_per_cap':lambda x: x,
#                   'Migrant_outflow_per_cap':lambda x: x,
                  'Net_immig_per_cap':lambda x: x,
                    'LFS_active_perc': lambda x: x
                  }

design_matrix = design_matrix.transform(transform_dict0)
design_matrix = design_matrix.astype(float)

interactions = [design_matrix.iloc[:,1:].multiply(region_dummies[dummy_name] ,axis = 0) for dummy_name in dummy_names]
# print(design_matrix.columns[1])
for idx, dummy_name in enumerate(dummy_names):
    interactions[idx] = interactions[idx].add_suffix('_' + dummy_name)
    

interactions = pd.concat(interactions,axis =1)
design_matrix = pd.concat([design_matrix, interactions],axis =1)
design_matrix = design_matrix.drop('LFS_active_perc',axis=1)
design_matrix = design_matrix.dropna()

results = sm.OLS(endog=design_matrix.iloc[:,0], exog=sm.add_constant(design_matrix.iloc[:,1:])).fit().summary()

# results = sm.OLS(endog=design_matrix.iloc[:,0], exog=design_matrix.iloc[:,1:]).fit().summary()

print(results)


# In[151]:


test5 = sm.OLS(endog=design_matrix.iloc[:,0], exog=sm.add_constant(design_matrix.iloc[:,1:])).fit()
test5.pvalues[test5.pvalues<0.05], test5.params[test5.pvalues<0.05]


# In[5]:



# %autoreload
# sys.path.append('C:/Users/Ben/Desktop/Python/Sublime/Reusable-Functions-and-other/')
# from Targeted_Functions import *
# test = Remove_Highly_Collinear_Variables(Pandas_Design_Matrix=design_matrix.iloc[:,1:],VIF_Threshold=10)


# In[6]:


outliers_influence.variance_inflation_factor(exog=sm.add_constant(design_matrix.iloc[:,1:]).values,exog_idx=[2])


# In[108]:


region_names = cross_section.Region.unique()
regional_cs_list = []
for reg_name in region_names:
    temp_cs = cross_section.loc[cross_section.Region == reg_name]
    regional_cs_list.append(temp_cs)

for idx, frame in enumerate(regional_cs_list):
    regional_cs_list[idx] = regional_cs_list[idx].loc[:,['Region','RegionName','AveragePrice','Population','dist_from_lon',
                                                      'Unemployment_rate','GDP_per_cap','Inflation_index','Inflation',
                                                      'Local_pshs','Outstanding_perc','Good_perc',
                                                      'Inad_perc','Non_UK_born','Migrant_inflow_monthly',
                                                      'Migrant_outflow_monthly','Net_immigration_monthly',
                                                      ]
                                                     ]
    regional_cs_list[idx].loc[:,'Real_GVA_per_cap'] = regional_cs_list[idx].loc[:,'GDP_per_cap'] *100/ regional_cs_list[idx].loc[:,'Inflation_index']
   
    cols_to_create = ['Migrant_inflow_per_cap','Migrant_outflow_per_cap','Net_immig_per_cap','Non_UK_per_cap','Local_pshs_per_cap']
    cols_to_use = ['Migrant_inflow_monthly','Migrant_outflow_monthly','Net_immigration_monthly','Non_UK_born','Local_pshs']
    
    for new, old in zip(cols_to_create, cols_to_use):
        regional_cs_list[idx][new] = regional_cs_list[idx][old]/regional_cs_list[idx]['Population']
for i in [9,10,11]:    
    for col in ['Outstanding_perc','Good_perc','Inad_perc', 'Non_UK_per_cap']:   
        regional_cs_list[i][col] = 0

matched_regions = [frame['Region'].max() for frame in regional_cs_list]
design_matrices = [frame.drop(['Region','RegionName','GDP_per_cap','Inflation_index', 'Migrant_inflow_monthly','Migrant_outflow_monthly','Net_immigration_monthly','Non_UK_born','Inflation','Local_pshs'],axis=1) for frame in regional_cs_list]
design_matrices = [dm.astype(float).dropna() for dm in design_matrices]

transform_dict = {'AveragePrice': np.log,
                  'dist_from_lon':np.log,
                  'Unemployment_rate': np.log,
                  'Real_GVA_per_cap':np.log,                                                      
                  'Local_pshs_per_cap':lambda x: np.log(x +1),
                  'Outstanding_perc':lambda x: np.log(x+1),
                  'Good_perc': lambda x: np.log(x+1),
                  'Inad_perc' : lambda x: np.log(x+1),
                  'Non_UK_per_cap':np.log,
                  'Migrant_inflow_per_cap':np.log,
                  'Migrant_outflow_per_cap':np.log,
                  'Net_immig_per_cap':lambda x: x
                  }

trans_design_matrices = list(design_matrices)
for idx, dm in enumerate(trans_design_matrices):
    trans_design_matrices[idx] = trans_design_matrices[idx].transform(transform_dict)
    trans_design_matrices[idx] = trans_design_matrices[idx].dropna()
    trans_design_matrices[idx]['GVA*unemp'] = trans_design_matrices[idx]['Real_GVA_per_cap']*trans_design_matrices[idx]['Unemployment_rate']
    trans_design_matrices[idx]['Immig*unemp'] = trans_design_matrices[idx]['Net_immig_per_cap']*trans_design_matrices[idx]['Unemployment_rate'] 
    trans_design_matrices[idx]['Inflow*unemp'] =trans_design_matrices[idx]['Migrant_inflow_per_cap']*trans_design_matrices[idx]['Unemployment_rate']
    trans_design_matrices[idx]['Outflow*unemp'] = trans_design_matrices[idx]['Migrant_outflow_per_cap']*trans_design_matrices[idx]['Unemployment_rate']


# In[149]:



exog_regressors = ['Real_GVA_per_cap',
                   'dist_from_lon',
                   'Unemployment_rate',                                                    
                   'Outstanding_perc',
                   'Good_perc',
                   'Inad_perc',
                   'Net_immig_per_cap',
                   'Non_UK_per_cap', 
                   'Local_pshs_per_cap' 
                   ]
endog_variable = 'AveragePrice'
for i in range(12):
    reg_num = i
    temp = trans_design_matrices[reg_num]
    if (i == 9)|(i==10)|(i==11):
        temp = temp.drop(['Outstanding_perc','Good_perc','Inad_perc', 'Non_UK_per_cap','Local_pshs_per_cap'],axis=1)
    temp=temp.dropna()
    endog = temp.loc[:,endog_variable]
    if i <9:
        exog = sm.add_constant(temp.loc[:,exog_regressors])
        results = sm.OLS(endog=endog, exog=exog).fit(cov_type='HC0')
    else:
        temp_regressors = ['Real_GVA_per_cap','dist_from_lon',
                           'Unemployment_rate','Net_immig_per_cap']
        exog = sm.add_constant(temp.loc[:,temp_regressors])
        results = sm.OLS(endog=endog, exog=exog).fit()
    print(str(i+1) +':',statsmodels.stats.diagnostic.het_breuschpagan(results.resid, exog_het = exog))
    


# In[118]:




