
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa as tssm
import os
import pyproj
import re
import time
import datetime
import gc
gc.enable()


# In[2]:


house_prices = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/House Prices/UK-HPI-full-file-2018-02.csv')

house_prices['Date'] = pd.to_datetime(house_prices['Date'], format = '%d/%m/%Y')
house_prices = house_prices.set_index('Date')
house_prices = house_prices[['RegionName', 'AreaCode','AveragePrice', 'Index',
                             'SalesVolume', 'DetachedPrice','SemiDetachedPrice','TerracedPrice','FlatPrice',
                             'CashPrice',
                             'CashSalesVolume',
                             'MortgageSalesVolume']]

loc_lookup1 = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Location/laregionlookup2012_tcm77-368555.xls', sheetname = [1],skiprows=4)[1]
loc_lookup2 = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Location/Local_Authority_District_to_Region_December_2016_Lookup_in_England.csv')

date_index = house_prices.index
loc_lookupdict1 = loc_lookup1.set_index('la_code')['region_name'].to_dict()
date_index = house_prices.index
loc_lookupdict2 = loc_lookup2.set_index('LAD16CD')['RGN16NM'].to_dict()
loc_lookupdict = {**loc_lookupdict1,**loc_lookupdict2}

nation_list = ['United Kingdom', 'England','Wales','Scotland','Northern Ireland', 'England and Wales']
nation_mask = house_prices['RegionName'].isin(['United Kingdom', 'England','Wales','Scotland','Northern Ireland'])

region_list = ['South East','East Midlands','North West',
               'South West', 'London','West Midlands', 'West Midlands Region',
               'Wales','Yorkshire and The Humber','North East','East of England','East']
region_mask = house_prices['RegionName'].isin(region_list)

lad_mask = ~ house_prices['RegionName'].isin(set(region_list+ nation_list))

house_price_lad  = house_prices[lad_mask]

house_price_lad['Region'] = house_price_lad['AreaCode'].replace(loc_lookupdict)

Eng_lad = house_price_lad[house_price_lad.Region.isin(region_list)]
temp_unasissgned_region = house_price_lad[~house_price_lad.isin(region_list)]
Scot_lad = temp_unasissgned_region[temp_unasissgned_region.Region.str[0] == 'S']
NI_lad = temp_unasissgned_region[temp_unasissgned_region.Region.str[0] == 'N']
Scot_lad.Region = Scot_lad.Region.map(lambda x: 'Scotland')
NI_lad.Region = NI_lad.Region.map(lambda x: 'Northern Ireland')

LAD_house_prices = pd.concat([Eng_lad,NI_lad,Scot_lad])
UK_house_prices = house_prices[house_prices['RegionName']=='United Kingdom']
Nation_house_prices = house_prices[nation_mask]
Region_house_prices = house_prices[region_mask]


# In[3]:


GVA = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/GDP/regionalgvaibylainuk.xls' , skiprows =2, sheetname= [2])[2]
population = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/GDP/regionalgvaibylainuk.xls' , sheetname =[3], skiprows=2)[3]

population = population.melt(id_vars = ['LAU1 code','LA name','Region'],value_vars=population.columns[5:], value_name='Population', var_name='Date')
GVA = GVA.melt(id_vars = ['LAU1 code','LA name','Region'],value_vars=GVA.columns[5:], value_name='GVA (millions)', var_name='Date')

population['Date'] = pd.to_datetime(population['Date'], format = '%Y')
population = population.set_index('Date')
GVA['Date'] = pd.to_datetime(GVA['Date'], format = '%Y')
GVA = GVA.set_index('Date')


# In[4]:


temp1 = LAD_house_prices.reset_index()
temp2= GVA.reset_index()
gva_lad = pd.merge(left = temp1, right=temp2, left_on = ['Date', 'AreaCode'], right_on = ['Date', 'LAU1 code'], how = 'left',suffixes=('', '_gva'))

temp1 =gva_lad
temp2= population.reset_index()
pop_gva_lad = pd.merge(left = temp1, right=temp2, left_on = ['Date', 'AreaCode'], right_on = ['Date', 'LAU1 code'], how = 'left',suffixes=('', '_pop'))
pop_gva_lad = pop_gva_lad.set_index('Date')
for lad_name in pop_gva_lad['RegionName'].unique():
        mask = pop_gva_lad['RegionName']==lad_name
        pop_gva_lad.loc[mask,['Population','GVA (millions)']] = pop_gva_lad.loc[mask,['Population','GVA (millions)']].fillna(method = 'ffill')
        
pop_gva_lad = pop_gva_lad.drop(['LAU1 code', 'LA name', 'Region_gva', 'LAU1 code_pop','LA name_pop', 'Region_pop'])


# In[5]:


combined_authority = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Location/LAD_to_LAU1_to_NUTS3_to_NUTS2_to_NUTS1_January_2018_Lookup_in_the_UK.csv')
nuts15_to18 =pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Location/NUTS_Level_3_2015_to_NUTS_Level_3_2018_Lookup_in_the_United_Kingdom.csv')

combined_authority = combined_authority.drop_duplicates(subset = 'LAD16NM')
temp_date_index = pop_gva_lad.index
pop_gva_lad_com = pd.merge_ordered(left = pop_gva_lad, right = combined_authority, left_on='RegionName', right_on = 'LAD16NM', how = 'left')
pop_gva_lad_com.index = temp_date_index



# In[27]:


pop_gva_lad_com.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/HPbyNUTS.csv')


# In[6]:


# listfiles = os.listdir("C:/Users/Ben/Downloads/Project/Q1/Location/codepo_gb/Data/CSV")
# pieces = []
# columns = ['pstcode','positional_quality_indicator','eastings','northings','country_code','nhs_regional_ha_code',
#            'nhs_ha_code','admin_county_code','admin_district_code','admin_ward_code']
# for f in listfiles:
#   path0 = "C:/Users/Ben/Downloads/Project/Q1/Location/codepo_gb/Data/CSV/%s" % f
#   frame=pd.read_csv(path0, names = columns)
#   frame['filename']=f
#   pieces.append(frame)
    
# postcodes = pd.concat(pieces, ignore_index=True)

# northings_dict = postcodes.set_index('admin_district_code')['northings'].to_dict()
# eastings_dict = postcodes.set_index('admin_district_code')['eastings'].to_dict()

# pop_gva_lad_com['northings'] = pop_gva_lad_com.AreaCode.replace(northings_dict)
# pop_gva_lad_com['eastings'] = pop_gva_lad_com.AreaCode.replace(eastings_dict)


# In[8]:


lat_long = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Location/ONS_Postcode_Directory_Latest_Centroids.csv')
lat_dict = lat_long.set_index('oslaua')['lat'].to_dict()
long_dict = lat_long.set_index('oslaua')['long'].to_dict()
osnorthing_dict = lat_long.set_index('oslaua')['oseast1m'].to_dict()
oseasting_dict = lat_long.set_index('oslaua')['osnrth1m'].to_dict()

del lat_long
pop_gva_lad_com['latitude'] = pop_gva_lad_com.AreaCode.replace(lat_dict)
pop_gva_lad_com['longitude'] = pop_gva_lad_com.AreaCode.replace(long_dict)
pop_gva_lad_com['os_northing'] = pop_gva_lad_com.AreaCode.replace(osnorthing_dict)
pop_gva_lad_com['os_easting'] = pop_gva_lad_com.AreaCode.replace(oseasting_dict)

for col in ['latitude', 'longitude','os_northing','os_easting']:
    pop_gva_lad_com[col] = pop_gva_lad_com[col].astype(float)

Lon_northing = float(531123.0)
Lon_easting = float(181374)
pop_gva_lad_com['dist_from_lon'] = ((pop_gva_lad_com['os_northing'] - Lon_northing)**2 + (pop_gva_lad_com['os_easting']-Lon_easting)**2)**0.5  
pop_gva_lad_com['dist_from_lon_index'] = pop_gva_lad_com['dist_from_lon'] *100/ pop_gva_lad_com['dist_from_lon'].max()


# In[10]:


unemp_lad = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Unemployment/unemp_lad.xlsx',skiprows = 7 ,sheetname = [0])[0]
unemp_lad = unemp_lad.iloc[1:1171]
unemp_lad = unemp_lad.melt(id_vars=['Area', 'mnemonic'], value_vars=unemp_lad.columns[2:],value_name='Unemployment', var_name='Date')
unemp_lad = unemp_lad.drop('Area', axis =1)
unemp_lad['Date'] = pd.to_datetime(unemp_lad['Date'], format = '%Y')


popnom_lad = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Population/pop_lad_nomis.csv',skiprows=7)
popnom_lad = popnom_lad.iloc[:-1,:]
colnames = popnom_lad.columns 
to_drop = colnames[colnames.str.contains('%')]
popnom_lad = popnom_lad.drop(to_drop, axis =1)
popnom_lad = popnom_lad.melt(id_vars=['Area', 'mnemonic'], value_vars=popnom_lad.columns[2:],value_name='Population', var_name='Date')
popnom_lad = popnom_lad.drop('Area', axis =1)
popnom_lad['Date'] = pd.to_datetime(popnom_lad['Date'], format = '%Y')

temp1 = pop_gva_lad_com.reset_index()
temp2 = unemp_lad.drop_duplicates(['Date','mnemonic'])

dataf = pd.merge(left = temp1, right = temp2, left_on=['Date', 'AreaCode'], right_on= ['Date', 'mnemonic'], how ='left',suffixes= ('', '_nomunemp'))
dataf = dataf.drop('mnemonic', axis =1)
dataf = dataf.replace({'-':np.nan})


#dataf['Unemployment'] = dataf['Unemployment'].fillna(method = 'ffill')
for lad_name in dataf['RegionName'].unique():
        mask = dataf['RegionName']==lad_name
        dataf.loc[mask,['Population','GVA (millions)','Unemployment']] = dataf.loc[mask,['Population','GVA (millions)','Unemployment']].fillna(method = 'ffill')

temp2 = popnom_lad.drop_duplicates(['Date','mnemonic'])
dataf = pd.merge(left = dataf, right = temp2, left_on=['Date', 'AreaCode'], right_on= ['Date', 'mnemonic'], how ='left',suffixes= ('', '_nom'))
dataf = dataf.drop('mnemonic', axis =1)
dataf = dataf.replace({'-':np.nan})

#dataf['Population_nom'] = dataf['Population_nom'].fillna(method = 'ffill')
for lad_name in dataf['RegionName'].unique():
        mask = dataf['RegionName']==lad_name
        dataf.loc[mask,['Population','GVA (millions)','Unemployment', 'Population_nom']] = dataf.loc[mask,['Population','GVA (millions)','Unemployment','Population_nom']].fillna(method = 'ffill')


# In[11]:


claimant = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Unemployment/claimant_count.csv', skiprows=8)
claimant = claimant.iloc[:797,:]
claimant = claimant.melt(id_vars = ['Area', 'mnemonic'], value_vars=claimant.columns[2:], var_name = 'Date', value_name='Claim_count')
claimant['Date']= pd.to_datetime(claimant['Date'], format = '%B %Y')
claimant = claimant.drop('Area', axis =1)
claimant = claimant.drop_duplicates()

dataf = pd.merge(left = dataf, right = claimant, left_on = ['Date', 'AreaCode'], right_on=['Date','mnemonic'], how ='left', suffixes = ('','_claim'))
dataf = dataf.drop('mnemonic', axis =1)
dataf = dataf.replace({'-':np.nan})
dataf['Claim_count'] = dataf['Claim_count'].astype(float)
dataf['Unemployment_rate'] = dataf['Claim_count']/dataf['Population']
dataf['GDP_per_cap'] = 1e6 * dataf['GVA (millions)']  / dataf['Population']
dataf = dataf.set_index('Date')


# In[12]:


inflation = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Inflation/consumerpriceinflationdetailedreferencetables.xls',sheetname = ['Table 48'], skiprows = 6)['Table 48']
inflation = inflation.iloc[1:73,3:]
inflation = inflation.replace({'  ..':np.nan,'..':np.nan})
inf_unrav = inflation.values.ravel()
temp_date_index = pd.DatetimeIndex(start='1947-01-01',end='2018-12-31', freq= 'M') - pd.offsets.MonthBegin(1)
inflation = pd.DataFrame(data=inf_unrav, index =temp_date_index)
inflation.columns=['Inflation_index']
inflation['Inflation'] = inflation['Inflation_index'].pct_change(1)
inflation.index.name = 'Date'
inflation = inflation.drop_duplicates()

temp_date_index = dataf.index
temp1= dataf.reset_index()
temp2 = inflation.reset_index()

df = pd.merge(left=temp1, right=temp2 ,left_on = 'Date', right_on = 'Date', how = 'left')
df= df.set_index('Date')


# In[13]:


money_supply = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/M4/M4_lending.csv',skiprows=9)
money_supply.head()

money_supply['LPMVQKW'][0]=0
m4_supply = 100  * money_supply['LPMVQKW'].div(100).add(1).cumprod()
money_supply['M4_supply'] = m4_supply

money_supply['DATE'] = pd.to_datetime(money_supply['DATE'], format = '%d %b %Y') - pd.offsets.MonthBegin(1)
money_supply = money_supply.set_index('DATE')
money_supply = money_supply[['LPMVQKW','M4_supply']]
money_supply.index.name = 'Date'
money_supply = money_supply.drop_duplicates()
money_supply.columns = ['M4_growth', 'M4_supply']

temp_date_index = dataf.index
temp1= df.reset_index()
temp2 = money_supply.reset_index()

dataf = pd.merge(left=temp1, right=temp2 ,left_on = 'Date', right_on = 'Date', how = 'left')
dataf.set_index('Date', inplace = True)


# In[14]:


real_yield = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Interest_rates/GLC Nominal month end data_1970 to 2015.xlsx',skiprows = 3,sheetname=['4. spot curve'])['4. spot curve']
nominal_yield = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Interest_rates/GLC Nominal month end data_1970 to 2015.xlsx',skiprows = 3, sheetname=['4. spot curve'])['4. spot curve']
blc_yield = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Interest_rates/BLC Nominal month end data_1990 to 2015.xlsx',skiprows = 3, sheetname= ['4. spot curve'])['4. spot curve']

real_yield = real_yield.iloc[1:,:]
nominal_yield = nominal_yield.iloc[1:,:]
blc_yield = blc_yield.iloc[1:,:]


real_yield['years:'] = real_yield['years:'] - pd.offsets.MonthBegin(1)
blc_yield['years:'] = blc_yield['years:'] - pd.offsets.MonthBegin(1)
nominal_yield['years:'] = nominal_yield['years:'] - pd.offsets.MonthBegin(1)
frame_list = [real_yield, blc_yield, nominal_yield]
for frame in frame_list:
    frame.columns = ['Date'] + frame.columns[1:].astype(str).tolist()
    


for idx, frame in enumerate(frame_list):
    frame['Yield_slope'] = frame['10'] - frame['1']
    frame.index =frame['Date']
    
temp_date_index = real_yield.index

Yield_slope = real_yield['Yield_slope'].to_frame()
Yield_slope = pd.merge(left = Yield_slope, right = blc_yield['Yield_slope'].to_frame(), left_index=True, right_index=True,how = 'outer',suffixes = ('_r','_blc'))
Yield_slope = pd.merge(left = Yield_slope, right = nominal_yield['Yield_slope'].to_frame(), left_index=True, right_index=True,how = 'outer',suffixes = ('','_n'))
Yield_slope.index = temp_date_index
Yield_slope = Yield_slope.drop_duplicates()

temp_date_index = dataf.index
temp1= dataf.reset_index()
temp2 = Yield_slope.reset_index()

df = pd.merge(left=temp1, right=temp2 ,left_on = 'Date', right_on = 'Date', how = 'left')
df.set_index('Date', inplace = True)


# In[15]:


xls = pd.ExcelFile('C:/Users/Ben/Downloads/Project/Q1/Housing_Starts/England-2005-onwards-by-local-authority.xlsx')
PSHS1= pd.read_excel(xls, skiprows = 3,sheetname = xls.sheet_names[:xls.sheet_names.index('2014 Q2')])
PSHS2= pd.read_excel(xls, skiprows = 4,sheetname = xls.sheet_names[xls.sheet_names.index('2014 Q2'):])


sheet_keys1 = list(PSHS1.keys())
sheet_keys2 = list(PSHS2.keys())
frames1 = [PSHS1[key].loc[5:445,['Current\nONS code','Lower and Single Tier Authority Data','Private\nEnterprise']] for key in sheet_keys1]
frames2 = [PSHS2[key].loc[5:4032,['Current\nONS code','Lower and Single Tier Authority Data','Private\nEnterprise']] for key in sheet_keys2]

date_range = (pd.date_range(start='2005-01-01', end='2017-12-01', freq='3M') - pd.offsets.MonthBegin(1)).tolist()

for frame, date in zip(frames1, date_range[:xls.sheet_names.index('2014 Q2')]):
    frame['Date'] = date

for frame, date in zip(frames2, date_range[xls.sheet_names.index('2014 Q2'):]):
    frame['Date'] = date

reduced_frames1 = [frame for frame in frames1]
reduced_frames2 = [frame for frame in frames2]


reduced_frames= reduced_frames1 + reduced_frames2

merged_pshs = pd.concat(reduced_frames, axis =0)
merged_pshs.columns = ['Current ONS code', 'Lower and Single Tier Authority Data','Local_pshs','Date']
merged_pshs = merged_pshs[['Current ONS code', 'Date','Local_pshs']]

temp_date_index = df.index
temp1= df.reset_index()
temp2 = merged_pshs

dataf = pd.merge(left=temp1, right=temp2 ,left_on = ['Date', 'AreaCode'], right_on = ['Date','Current ONS code'], how = 'left')
dataf.set_index('Date', inplace = True)

for lad_code in dataf.loc[:'2017-12-31','AreaCode'].unique():
    dataf.loc[lambda d: (d.index<'2018')&(d.index>'2004')&(d['AreaCode'] == lad_code),['Local_pshs']] = dataf.loc[lambda d: (d.index<'2018')&(d.index>'2004')&(d.AreaCode == lad_code),['Local_pshs']].fillna(method = 'ffill')

dataf['Local_pshs_monthly'] = dataf['Local_pshs']/3


# In[16]:


PSHS_pre12 = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Housing_Starts/Housing-starts-in-England-pre-2012.xls', skiprows = 4,sheetname = 0)
PSHS_NI = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Housing_Starts/Housing-starts-in-NI.xlsx', skiprows = 3,sheetname = 0)
PSHS_Scot =  pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Housing_Starts/Housing-starts-in-Scotland.xlsx', skiprows = 3,sheetname = 0)
PSHS_Wal =  pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Housing_Starts/Housing-starts-in-Wales.xlsx', skiprows = 3,sheetname = 0)

frames =  [PSHS_NI, PSHS_Wal, PSHS_Scot]
mapping_dict = {'Q1':'01-01','Q2':'04-01','Q3':'07-01','Q4':'10-01'}
countries = ['Northern Ireland', 'Wales', 'Scotland']
for idx, frame in enumerate(frames):
    frames[idx] = frame.iloc[1:200, [0,1,4]]
    frames[idx].columns = ['Year', 'Quarter', 'Private Enterprise']
    frames[idx]['Year'].fillna(method='ffill',inplace = True)
    frames[idx] = frames[idx][frames[idx]['Quarter'].notnull()]
    frames[idx]['Month'] = frames[idx]['Quarter'].map(mapping_dict)
    frames[idx]['Year'] = frames[idx]['Year'].astype(str)
    frames[idx]['Date'] = frames[idx]['Year'] +'-' + frames[idx]['Month']
    frames[idx]['Date'] = pd.to_datetime(frames[idx]['Date'])
    frames[idx] = frames[idx][['Date', 'Private Enterprise']]
    frames[idx]['Country'] = countries[idx]

merged_country_pshs = pd.concat(frames, axis = 0)


pre12 = PSHS_pre12.copy()
pre12 = pre12.iloc[1:1026,[0,1,4]]
pre12.columns = ['Year', 'Quarter', 'Private Enterprise']
pre12['Year'].fillna(method='ffill',inplace = True)
pre12['Month'] = pre12['Quarter'].map(mapping_dict)




pre12['Region'] =  pre12['Year'].map(lambda x: x if re.match(pattern='[A-Za-z]+',string=str(x)) else np.nan)
pre12['Region'] = pre12['Region'].fillna(method = 'ffill')
mask = ~pre12['Year'].str.match('[A-Za-z]+').fillna(False)
pre12['Year'].astype(str,inplace = True)
pre12 = pre12[mask].dropna()
pre12['Date'] = pd.to_datetime(pre12['Year'].astype(str,inplace=True) + '-' + pre12['Month'])

pre12 = pre12[['Date', 'Region', 'Private Enterprise']]
pre12['Region'] = pre12['Region'].str.strip()
pre12['Region'] = pre12['Region'].replace({'Yorkshire and the Humber':'Yorkshire and The Humber',})

merged_country_pshs.columns = ['Date','Private Enterprise', 'Region']

pre12 = PSHS_pre12.copy()
pre12 = pre12.iloc[1:1026,[0,1,4]]
pre12.columns = ['Year', 'Quarter', 'Private Enterprise']
pre12['Year'].fillna(method='ffill',inplace = True)
pre12['Month'] = pre12['Quarter'].map(mapping_dict)




pre12['Region'] =  pre12['Year'].map(lambda x: x if re.match(pattern='[A-Za-z]+',string=str(x)) else np.nan)
pre12['Region'] = pre12['Region'].fillna(method = 'ffill')
mask = ~pre12['Year'].str.match('[A-Za-z]+').fillna(False)
pre12['Year'].astype(str,inplace = True)
pre12 = pre12[mask].dropna()
pre12['Date'] = pd.to_datetime(pre12['Year'].astype(str,inplace=True) + '-' + pre12['Month'])

pre12 = pre12[['Date', 'Region', 'Private Enterprise']]
pre12['Region'] = pre12['Region'].str.strip()
pre12['Region'] = pre12['Region'].replace({'Yorkshire and the Humber':'Yorkshire and The Humber',})

merged_country_pshs.columns = ['Date','Private Enterprise', 'Region']

merged_reg_pshs = pd.concat([merged_country_pshs, pre12],axis = 0)
full_merged_pshs = pd.concat([merged_pshs, merged_reg_pshs],axis = 0)

mask = full_merged_pshs['Current ONS code'].notnull()
full_merged_pshs.loc[mask,'Region'] = full_merged_pshs.loc[mask,'Current ONS code'].replace(loc_lookupdict)
full_merged_pshs = full_merged_pshs[full_merged_pshs['Region'].notnull()]
full_merged_pshs = full_merged_pshs.drop_duplicates()
full_merged_pshs= full_merged_pshs.replace({'  ..':np.nan,'..':np.nan})
full_merged_pshs['Local_pshs'] = full_merged_pshs['Local_pshs'].astype(float)
full_merged_pshs['Private Enterprise'] = full_merged_pshs['Private Enterprise'].astype(float)

pshs_region_tot = full_merged_pshs.groupby(['Date','Region']).sum().loc[:,'Private Enterprise']
pshs_region_tot = pshs_region_tot.reset_index(level='Region')
pshs_region_tot.columns = ['Region', 'Regional_PSHS']
pshs_region_tot = pshs_region_tot.replace({'  ..':np.nan,'..':np.nan})
pshs_region_tot['Region'] = pshs_region_tot['Region'].replace({'East': 'East of England'})

temp_date_index = dataf.index
temp1= dataf.reset_index()
temp2 = pshs_region_tot.reset_index()

df = pd.merge(left=temp1, right=temp2 ,left_on = ['Date', 'Region'], right_on = ['Date','Region'], how = 'left')
df.set_index('Date', inplace = True)

for reg_name in df['Region'].unique():
        mask = df['Region']==reg_name
        df.loc[mask,['Regional_PSHS']] =df.loc[mask,['Regional_PSHS']].fillna(method = 'ffill')
df.loc['2018','Regional_PSHS'] = np.nan


# In[19]:


xls = pd.ExcelFile('C:/Users/Ben/Downloads/Project/Q1/Education/3.SchoolQualityTables_Jan.xlsx')
education = pd.read_excel(xls, sheetname='Table SQ8', skiprows = 7)
education = education.iloc[:364, [0,1,14,15,17]]

education.columns = ['LAD_Code', 'Area_Name','Outstanding_perc','Good_perc','Inad_perc']
education = education[['LAD_Code','Outstanding_perc','Good_perc','Inad_perc']]

temp_date_index = df.index
temp1= df.reset_index()
temp2 = education

df = pd.merge(left=temp1, right=temp2 ,left_on ='AreaCode', right_on = 'LAD_Code', how = 'left')
df.set_index('Date', inplace = True)


for lad_name in df['RegionName'].unique():
        mask = df['RegionName']==lad_name
        df.loc[mask,['Outstanding_perc','Good_perc','Inad_perc']] = df.loc[mask,['Outstanding_perc','Good_perc','Inad_perc']].fillna(method = 'ffill').fillna(method ='bfill')


# In[20]:


mortgage = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Mortgage/Mortgage-rate-information-BoE.csv',skiprows = 11)
mortgage.columns =['Date', '2yr_95ltv','5yr_95ltv', '2yr_90ltv','2yr_75ltv','3yr_75_ltv','5yr_75ltv','2yr-60ltv','2yr_85ltv','2yr_75ltvbtl']
mortgage['Date']=pd.to_datetime(mortgage['Date'],format = '%d-%b-%y') - pd.offsets.MonthBegin(1)
mortgage = mortgage.replace({'  ..':np.nan,'..':np.nan})

temp_date_index = df.index
temp1= df.reset_index()
temp2 = mortgage

dataf = pd.merge(left=temp1, right=temp2 ,on = 'Date', how = 'left')
dataf.set_index('Date', inplace = True)



# In[21]:


xls = pd.ExcelFile('C:/Users/Ben/Downloads/Project/Q1/Immigration/immigration-data.xls')
immig = pd.read_excel(xls,sheetname = [1,3], skiprows = 0)

cols_mask_in = [0,1] + [(5*i +3) for i in range(12)]
cols_mask_out = [0,1] + [(5*i +4) for i in range(12)]
inflows = immig[1].iloc[3:443,cols_mask_in ]
outflows = immig[1].iloc[3:443,cols_mask_out]

date_range = pd.date_range(start = '2005' ,end = '2017', freq = 'A') - pd.offsets.YearBegin(1)

inflows.columns = ['Area_Code', 'Area_Name'] + date_range.tolist()
outflows.columns = ['Area_Code', 'Area_Name'] + date_range.tolist()

inflows = inflows.melt(id_vars=['Area_Code','Area_Name'], value_vars=inflows.columns[2:],value_name='Migrant_inflow_mid',var_name='Date')
outflows = outflows.melt(id_vars=['Area_Code','Area_Name'], value_vars=outflows.columns[2:],value_name='Migrant_outflow_mid', var_name='Date')

inflows = inflows[inflows.Area_Code.notnull()]
outflows = outflows[outflows.Area_Code.notnull()]



inflowsp1m = inflows.copy()
outflowsp1m = outflows.copy()

inflowsp1m['Date'] = inflowsp1m['Date'] + pd.offsets.YearBegin(1)
outflowsp1m ['Date'] = outflowsp1m['Date'] + pd.offsets.YearBegin(1)


inflowsmer = pd.merge(left = inflows, right = inflowsp1m, left_on=['Date', 'Area_Code'],right_on = ['Date','Area_Code'],how='outer', suffixes= ('','_1'))
outflowsmer = pd.merge(left = outflows, right = outflowsp1m, left_on=['Date', 'Area_Code'],right_on = ['Date','Area_Code'],how='outer', suffixes= ('','_1'))

inflowsmer['Migrant_inflow'] = 0.5 * (inflowsmer['Migrant_inflow_mid'] + inflowsmer['Migrant_inflow_mid_1'])
outflowsmer['Migrant_outflow'] = 0.5 * (outflowsmer['Migrant_outflow_mid'] + outflowsmer['Migrant_outflow_mid_1'])

inflows = inflowsmer[['Date','Area_Code','Migrant_inflow']].dropna()
outflows = outflowsmer[['Date','Area_Code','Migrant_outflow']].dropna()

migflows = pd.merge(left = inflows, right=outflows, left_on=['Date', 'Area_Code'],right_on = ['Date','Area_Code'],how='inner')
migflows['Net_immigration'] = migflows.Migrant_inflow - migflows.Migrant_outflow


migflows.head()

non_uk = immig[3].copy()
cols_mask = [0,1] + [3*i + 3 for i in range(12)]
non_uk = non_uk.iloc[3:,cols_mask]

date_range = pd.date_range(start = '2005' ,end = '2017', freq = 'A') - pd.offsets.YearBegin(1)
non_uk.columns = ['Area_Code', 'Area_Name'] + date_range.tolist()

non_uk = non_uk.melt(id_vars=['Area_Code','Area_Name'], value_vars=non_uk.columns[2:],value_name='Non_UK_born',var_name='Date')

non_uk = non_uk[non_uk['Area_Code'].notnull()]
non_uk = non_uk.drop('Area_Name',axis =1)


Immigration = pd.merge(left = migflows, right = non_uk, left_on=['Date', 'Area_Code'],right_on = ['Date','Area_Code'],how='outer')

area_list = []
for idx, area in enumerate(Immigration.Area_Code.unique()):
    area_list.append(Immigration[Immigration.Area_Code ==area])
    area_list[idx] = area_list[idx].set_index('Date').resample('MS').ffill()
Immigration = pd.concat(area_list, axis = 0)

temp_date_index = dataf.index
temp1= dataf.reset_index()
temp2 = Immigration.reset_index()

df = pd.merge(left=temp1, right=temp2 ,left_on=['Date', 'AreaCode'],right_on = ['Date','Area_Code'], how = 'left')
df.set_index('Date', inplace = True)

for region_name in df.loc[:'2016-12-31','Region'].unique():
    df.loc[lambda d: (d.index<'2017')&(d.Region == region_name),['Migrant_inflow','Migrant_outflow','Net_immigration','Non_UK_born']] = df.loc[lambda d: (d.index<'2017')&(d.Region == region_name),['Migrant_inflow','Migrant_outflow','Net_immigration','Non_UK_born']].fillna(method = 'ffill')

df[['Migrant_inflow_monthly','Migrant_outflow_monthly','Net_immigration_monthly']] = df[['Migrant_inflow','Migrant_outflow','Net_immigration']]/12


# In[22]:


curracc = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Current Account/Current-account-balance.csv',skiprows=7)
curracc.columns = ['Year','Cur_def_perc_gdp']
curracc = curracc.iloc[31:,:]
curracc['Date'] = pd.date_range(start= '1987-01-01', end = '2018-01-01',freq = '3M') - pd.offsets.MonthBegin(1)
curracc = curracc[['Date','Cur_def_perc_gdp']]

temp_date_index = df.index
temp1= df.reset_index()
temp2 = curracc

dataf = pd.merge(left=temp1, right=temp2 ,on='Date', how = 'left')
dataf.set_index('Date', inplace = True)

dataf.loc[:'2017-12-31', 'Cur_def_perc_gdp'] = dataf.loc[:'2017-12-31', 'Cur_def_perc_gdp'].fillna(method = 'ffill')


gc.collect()


# In[23]:


bud_def =  pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Deficit/Budget_deficit.csv',skiprows = 1).iloc[:21,:]
bud_def.columns = ['Year', 'link',' pop', 'Budget_deficit', 'a']
bud_def['Date'] = pd.to_datetime(bud_def['Year'].astype(str), format = '%Y')
bud_def = bud_def[['Date','Budget_deficit']]

temp_date_index = dataf.index
temp1= dataf.reset_index()
temp2 = bud_def

df = pd.merge(left=temp1, right=temp2 ,on='Date', how = 'left')
df.set_index('Date', inplace = True)

df.loc['1997-01-01':'2017-12-31', 'Budget_deficit'] = df.loc['1997-01-01':'2017-12-31', 'Budget_deficit'].fillna(method = 'ffill')


# In[24]:


bank_rate = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/Standard_lending/Std_lending_rate.csv',skiprows =2)
bank_rate.columns = ['Date', 'Bank_rate']
bank_rate['Date'] = pd.to_datetime(bank_rate['Date'], format = '%d %b %Y') - pd.offsets.MonthBegin(1)
bank_rate = bank_rate.drop_duplicates()
bank_rate = bank_rate.groupby('Date').mean().reset_index()

mask = bank_rate['Bank_rate'].diff().round()!=0
bank_rate['Bank_rate_change_indicator'] = 0
bank_rate.loc[mask, 'Bank_rate_change_indicator']=1


temp_date_index = df.index
temp1= df.reset_index()
temp2 = bank_rate

dataf = pd.merge(left=temp1, right=temp2 ,on='Date', how = 'left')
dataf.set_index('Date', inplace = True)

dataf['Bank_rate'] = dataf['Bank_rate'].fillna(method ='ffill')


gc.collect()


# In[22]:


econ_pop =  pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Unemployment/Econ_active.xlsx',skiprows =7).iloc[:310]
econ_active = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Unemployment/Econ_active.xlsx',skiprows =7).iloc[322:632]
econ_emp =  pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Unemployment/Econ_active.xlsx',skiprows =7).iloc[644:954]
econ_unemp =  pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/Unemployment/Econ_active.xlsx',skiprows =7).iloc[966:1276]

frames = [econ_pop, econ_active, econ_emp, econ_unemp]
measures = ['LFS_population', 'LFS_active', 'LFS_employed','LFS_unemployed']

for idx, frame in enumerate(frames):
    frames[idx]['Date'] = frame['Date'].str.split('-',expand=True)[0]
    frames[idx]['Date'] = pd.to_datetime(frames[idx]['Date'],format = '%b %Y')
    frames[idx] = frames[idx].melt(id_vars = ['Date'], value_vars=frames[idx].columns[1:], var_name = 'Region', value_name = measures[idx])

LFS_merged = pd.concat(frames, axis =1).iloc[:,[0,1,2,5,8,11]]
LFS_merged['LFS_active_perc'] = LFS_merged['LFS_active'] / LFS_merged['LFS_population']
LFS_merged['LFS_unemployed_perc'] = LFS_merged['LFS_unemployed'] / LFS_merged['LFS_active']
LFS_merged['Region'] = LFS_merged['Region'].replace({'East':'East of England'})

temp_date_index = dataf.index
temp1= dataf.reset_index()
temp2 = LFS_merged

df = pd.merge(left=temp1, right=temp2 ,left_on=['Date','Region'], right_on = ['Date','Region'],how = 'left', suffixes = ('', '_LFS'))
df.set_index('Date', inplace = True)

columns_to_drop = ['CashPrice','CashSalesVolume', 'MortgageSalesVolume','LAU1 code', 'LA name',
                   'Region_gva','LAU1 code_pop','LA name_pop', 'Region_pop','LAD16CD', 'LAD16NM',
                   'LAU118CD', 'LAU118NM', 'NUTS318CD', 'NUTS318NM', 'NUTS218CD',
                   'NUTS218NM', 'NUTS118CD', 'NUTS118NM', 'FID', 'latitude', 'longitude',
                   'os_northing', 'os_easting', 'Current ONS code','LAD_Code','Area_Code'
                  ]
df = df.drop(columns_to_drop,axis =1)

full_data = df.copy()
full_data = full_data.replace({'  ..':np.nan,'..':np.nan,'.':np.nan,'c':np.nan,':':np.nan})
full_data = full_data.reset_index().drop_duplicates(subset = ['Date','AreaCode','RegionName']).set_index('Date')
mapping_func = lambda x: 1 if x in (1,2,3) else 2 if x in (4,5,6) else 3 if x in (7,8,9) else 4 if x in (10,11,12) else 0
full_data['Quarter_indicator'] = full_data.index.month.map(mapping_func).values
full_data['GVA_diff'] = full_data['GVA (millions)'].diff(1)
Regional_population = full_data.groupby(by = 'Region')['Population'].sum().to_dict()
full_data['Regional_population']= full_data['Region'].map(Regional_population)

non_eng_regions =['Wales','Scotland','Northern Ireland']
for reg in non_eng_regions:
    mask = full_data['Region'] ==reg
    full_data.loc[mask,'local_pshs'] = full_data.loc[mask,'Regional_PSHS'] * full_data.loc[mask,'Population']/ full_data.loc[mask,'Regional_population']


full_data.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Full_LAD_Data.csv')


# In[23]:


agg_dict_quarterly = {'RegionName':'first','AreaCode':'first', 'AveragePrice':'mean','Index':'mean',
             'SalesVolume':'sum','DetachedPrice':'mean','SemiDetachedPrice':'mean',
             'TerracedPrice':'mean','FlatPrice':'mean','Region':'first','GVA (millions)':'mean',
            'Population':'mean','dist_from_lon':'first','dist_from_lon_index':'first',
             'Unemployment':'mean','Population_nom':'mean','Claim_count':'mean',
             'Unemployment_rate':'mean','GDP_per_cap':'mean','Inflation_index':'mean','Inflation':'mean',
             'M4_growth':'sum','M4_supply':'mean','Yield_slope_r':'mean','Yield_slope_blc':'mean',
             'Yield_slope':'mean','Local_pshs':'mean','Local_pshs_monthly':'sum','Regional_PSHS':'mean',
             'Outstanding_perc':'mean', 'Good_perc':'mean','Inad_perc':'mean','2yr_95ltv':'mean',
             '5yr_95ltv':'mean','2yr_90ltv':'mean','2yr_75ltv':'mean','3yr_75_ltv':'mean',
             '5yr_75ltv':'mean','2yr-60ltv':'mean','2yr_85ltv':'mean','2yr_75ltvbtl':'mean',
             'Migrant_inflow':'mean','Migrant_outflow':'mean','Net_immigration':'mean',
             'Non_UK_born':'mean', 'Migrant_inflow_monthly':'sum','Migrant_outflow_monthly':'sum',
             'Net_immigration_monthly':'sum','Cur_def_perc_gdp':'mean','Budget_deficit':'mean','Bank_rate':'mean',
             'Bank_rate_change_indicator':'max', 'Regional_population':'mean',
             'LFS_population':'mean','LFS_active':'mean','LFS_employed':'mean','LFS_unemployed':'mean',
             'LFS_active_perc':'mean','LFS_unemployed_perc':'mean','Quarter_indicator':'first',
             'GVA_diff':'sum'
            }
numeric_columns = full_data.columns[2:][full_data.columns[2:]!='Region']
categorical_columns = ['AreaCode','RegionName','Region']


quarterly_data = full_data.copy()

for col in numeric_columns:
    quarterly_data[col]=quarterly_data[col].astype(float)
for col in categorical_columns:
    quarterly_data[col]=quarterly_data[col].astype('category')
    
quarterly_data = quarterly_data.groupby('AreaCode').resample('Q').agg(agg_dict_quarterly)
quarterly_data['Quarter_indicator'] = quarterly_data.reset_index(level = 'AreaCode',drop =True).index.month.map(mapping_func).values
quarterly_data = quarterly_data.reset_index(level =0, drop = True)

quarterly_data.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Quarterly_Data.csv')


# In[24]:


agg_dict_yearly = {'RegionName':'first','AreaCode':'first', 'AveragePrice':'mean','Index':'mean',
             'SalesVolume':'sum','DetachedPrice':'mean','SemiDetachedPrice':'mean',
             'TerracedPrice':'mean','FlatPrice':'mean','Region':'first','GVA (millions)':'mean',
            'Population':'mean','dist_from_lon':'first','dist_from_lon_index':'first',
             'Unemployment':'mean','Population_nom':'mean','Claim_count':'mean',
             'Unemployment_rate':'mean','GDP_per_cap':'mean','Inflation_index':'mean','Inflation':'mean',
             'M4_growth':'sum','M4_supply':'mean','Yield_slope_r':'mean','Yield_slope_blc':'mean',
             'Yield_slope':'mean','Local_pshs':lambda x: x.sum()/4,'Local_pshs_monthly':'sum','Regional_PSHS':'mean',
             'Outstanding_perc':'mean', 'Good_perc':'mean','Inad_perc':'mean','2yr_95ltv':'mean',
             '5yr_95ltv':'mean','2yr_90ltv':'mean','2yr_75ltv':'mean','3yr_75_ltv':'mean',
             '5yr_75ltv':'mean','2yr-60ltv':'mean','2yr_85ltv':'mean','2yr_75ltvbtl':'mean',
             'Migrant_inflow':'mean','Migrant_outflow':'mean','Net_immigration':'mean',
             'Non_UK_born':'mean', 'Migrant_inflow_monthly':'sum','Migrant_outflow_monthly':'sum',
             'Net_immigration_monthly':'sum','Cur_def_perc_gdp':'mean','Budget_deficit':'mean','Bank_rate':'mean',
             'Bank_rate_change_indicator':'max', 'Regional_population':'mean',
             'LFS_population':'mean','LFS_active':'mean','LFS_employed':'mean','LFS_unemployed':'mean',
             'LFS_active_perc':'mean','LFS_unemployed_perc':'mean','Quarter_indicator':'first',
             'GVA_diff':'sum'
            }

yearly_data = full_data.copy()

for col in numeric_columns:
    yearly_data[col]=yearly_data[col].astype(float)
for col in categorical_columns:
    yearly_data[col]=yearly_data[col].astype('category')
    
yearly_data = yearly_data.groupby('AreaCode').resample('A').agg(agg_dict_yearly)
yearly_data['Quarter_indicator'] = yearly_data.reset_index(level = 'AreaCode',drop =True).index.month.map(mapping_func).values

indexer = pd.IndexSlice

yearly_data = yearly_data.reset_index(level = 0, drop = True)

yearly_data.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Annual_Data.csv')



# In[25]:


nationwide = pd.read_excel('C:/Users/Ben/Downloads/Project/Q1/House Prices/F1_House_prices_UK_Nationwide.xlsx',sheetname=1)
nationwide = nationwide.iloc[2:,:]
cols_mask_price = [0] + [(2*i +1) for i in range(13)]
cols_mask_index = [0] + [(2*i +2) for i in range(13)]


nationwide_prices = nationwide[nationwide.columns[cols_mask_price]]
nationwide_index= nationwide[nationwide.columns[cols_mask_index]]

nationwide_prices['Greater London'] = 0.5*(nationwide_prices.iloc[:,8] + nationwide_prices.iloc[:,9] )
nationwide_index['Greater London'] = 0.5*(nationwide_index.iloc[:,8] + nationwide_index.iloc[:,9] )

nationwide_prices = nationwide_prices.melt(id_vars = ['Q1  1993'],value_vars=nationwide_prices.columns[1:], value_name='Nationwide_average_hprice_region', var_name='REGION')
nationwide_index = nationwide_index.melt(id_vars = ['Q1  1993'],value_vars=nationwide_index.columns[1:], value_name='Nationwide_index_region', var_name='REGION')

mapping_dict = {'NORTH':'North East', 'YORKS & HSIDE':'Yorkshire and The Humber', 'NORTH WEST':'North West',
                'EAST MIDS':'East Midlands',
                'WEST MIDS':'West Midlands', 'EAST ANGLIA':'East of England', 'OUTER S EAST':'South East',
                'OUTER MET':np.nan, 'LONDON':np.nan,
               'SOUTH WEST':'South West', 'WALES':'Wales', 'SCOTLAND':'Scotland', 'N IRELAND':'Northern Ireland',
                'Greater London':'London'}
nationwide_prices['Region'] = nationwide_prices['REGION'].map(mapping_dict)
nationwide_index['Region'] = nationwide_prices['Region']

nationwide_prices = nationwide_prices.iloc[:,[0,3,2]]
nationwide_index = nationwide_index.iloc[:,[0,3,2]]

nationwide_prices.columns = ['Date','Region','Nationwide_average_hprice_region']
nationwide_index.columns = ['Date','Region','Nationwide_index_region']

mapping_dict = {'Q1':'01-01','Q2':'04-01','Q3':'07-01','Q4':'10-01'}

nationwide_prices['Year'] = nationwide_prices['Date'].str.split(' ',expand=True)[1]
nationwide_prices['Month'] = nationwide_prices['Date'].str.split(' ',expand=True)[0].map(mapping_dict)
nationwide_prices['Date'] = nationwide_prices['Year'] + '-' + nationwide_prices['Month'] 

nationwide_index['Year'] = nationwide_index['Date'].str.split(' ',expand=True)[1]
nationwide_index['Month'] = nationwide_index['Date'].str.split(' ',expand=True)[0].map(mapping_dict)
nationwide_index['Date'] = nationwide_index['Year'] + '-' + nationwide_index['Month'] 

nationwide_prices = nationwide_prices.iloc[:,[0,1,2]]
nationwide_index = nationwide_index.iloc[:,[0,1,2]]

nationwide = pd.merge(left = nationwide_prices, right = nationwide_index, left_on = ['Date','Region'], right_on =['Date','Region'] , how = 'inner')
nationwide['Date'] = pd.to_datetime(nationwide['Date'],format = '%Y-%m-%d')
nationwide['Nationwide_average_hprice_region'] = nationwide['Nationwide_average_hprice_region'].astype(float)
nationwide['Nationwide_index_region'] = nationwide['Nationwide_average_hprice_region'].astype(float)
nationwide['Region'] = nationwide['Region'].astype('category')


# In[26]:


nationwide15 = nationwide.set_index('Date')['2015'].groupby('Region').resample('A').agg({'Nationwide_average_hprice_region':'mean','Nationwide_index_region':'mean'})
nationwide15 = nationwide15.reset_index(level = 0)
twenty_15_data = yearly_data['2015']
twenty_15_data['Date'] = pd.datetime(year = 2015, month = 12,day = 31)
twenty_15_data = twenty_15_data.reset_index(level=0,drop =True)

temp_date_index = twenty_15_data.index
temp1= twenty_15_data
temp2 = nationwide15.reset_index()

twenty_15_data = pd.merge(left=temp1, right=temp2 ,left_on=['Date','Region'], right_on = ['Date','Region'],how = 'left', suffixes = ('', '_nat'))
twenty_15_data.set_index('Date', inplace = True)

twenty_15_data.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Cross_Section_Data.csv')


# In[27]:


agg_dict_region = {'AveragePrice':'mean','Index':'mean',
             'SalesVolume':'sum','DetachedPrice':'mean','SemiDetachedPrice':'mean',
             'TerracedPrice':'mean','FlatPrice':'mean','GVA (millions)':'sum',
            'Population':'sum','dist_from_lon':'mean','dist_from_lon_index':'mean',
             'Unemployment':'mean','Population_nom':'sum','Claim_count':'sum',
             'Unemployment_rate':'mean','GDP_per_cap':'mean','Inflation_index':'mean','Inflation':'mean',
             'M4_growth':'mean','M4_supply':'mean','Yield_slope_r':'mean','Yield_slope_blc':'mean',
             'Yield_slope':'mean','Local_pshs':'sum','Local_pshs_monthly':'sum','Regional_PSHS':'mean',
             'Outstanding_perc':'mean', 'Good_perc':'mean','Inad_perc':'mean','2yr_95ltv':'mean',
             '5yr_95ltv':'mean','2yr_90ltv':'mean','2yr_75ltv':'mean','3yr_75_ltv':'mean',
             '5yr_75ltv':'mean','2yr-60ltv':'mean','2yr_85ltv':'mean','2yr_75ltvbtl':'mean',
             'Migrant_inflow':'sum','Migrant_outflow':'sum','Net_immigration':'sum',
             'Non_UK_born':'sum', 'Migrant_inflow_monthly':'sum','Migrant_outflow_monthly':'sum',
             'Net_immigration_monthly':'sum','Cur_def_perc_gdp':'mean','Budget_deficit':'mean','Bank_rate':'mean',
             'Bank_rate_change_indicator':'max', 'Regional_population':'mean',
             'LFS_population':'mean','LFS_active':'mean','LFS_employed':'mean','LFS_unemployed':'mean',
             'LFS_active_perc':'mean','LFS_unemployed_perc':'mean','Quarter_indicator':'first',
             'GVA_diff':'sum'
            }
ts_region_data0 = quarterly_data.reset_index().groupby(['Region','Date']).agg(agg_dict_region)


ts_region_data1 = quarterly_data.groupby('Region').resample('Q').agg(agg_dict_region)

ts_region_data0 = ts_region_data0.reset_index(level=0)
nationwide['Date'] = nationwide['Date'] + pd.offsets.QuarterEnd(1)

temp_date_index = ts_region_data0.index
temp1= ts_region_data0.reset_index()
temp2 = nationwide

ts_region_data0 = pd.merge(left=temp1, right=temp2 ,left_on=['Date','Region'], right_on = ['Date','Region'],how = 'left', suffixes = ('', '_nat'))
ts_region_data0.set_index('Date', inplace = True)


ts_region_data0.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Time_Series_Data.csv')


# In[28]:


temp = full_data.copy()

numeric_columns = full_data.columns[2:][full_data.columns[2:]!='Region']
categorical_columns = ['AreaCode','RegionName','Region']

for col in numeric_columns:
    temp[col]=temp[col].astype(float)
for col in categorical_columns:
    temp[col]=temp[col].astype('category')
full_regional_data = temp.reset_index().groupby(['Region','Date']).agg(agg_dict_region)
full_regional_data = full_regional_data.reset_index(level=0)
full_regional_data.to_csv('C:/Users/Ben/Downloads/Project/Q1/Full_data/Full_Regional_Data.csv')


# In[30]:


import requests
import json


test4 = requests.get('https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/gb/lad.json').json()
with open('C:/Users/Ben/Downloads/Project/Q1/Location/GBgeo.json','w') as outfile:
    json.dump(test4, outfile)
    
test5 = requests.get('https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/ni/lgd.json').json()
with open('C:/Users/Ben/Downloads/Project/Q1/Location/IRLgeo.json','w') as outfile:
    json.dump(test5, outfile)


# In[31]:


import pandas as pd
import os
import json

# read in population data
df = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/House Prices/UK-HPI-full-file-2018-02.csv')
df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
df = df.set_index('Date')
reduced95 = df['1995'].groupby('RegionName').mean().reset_index()
reduced95['log_avg_price'] = np.log(reduced95['AveragePrice'])
reduced95['RankPrice'] = reduced95['AveragePrice'].rank(pct=True,ascending=False)*100

from branca.utilities import split_six
quantile, scale = pd.qcut(reduced95['RankPrice'], retbins=True, q = 6 )
import folium
state_geo1 = os.path.join('data', 'C:/Users/Ben/Downloads/Project/Q1/Location/GBgeo.json')
state_geo2 = os.path.join('data', 'C:/Users/Ben/Downloads/Project/Q1/Location/IRLgeo.json')

m = folium.Map(location=[55, 4], zoom_start=5, tiles = 'cartodbpositron')
m.choropleth(
    geo_data=state_geo1,
    data=reduced95,
    columns=['RegionName', 'RankPrice'],
    key_on = 'feature.properties.LAD13NM',
    fill_color='RdYlBu',
    fill_opacity=0.9,
    line_opacity=0.2,
    legend_name='Rank of Average House Price',
    highlight=True,
    
)

m.choropleth(
    geo_data=state_geo2,
    data=reduced95,
    columns=['RegionName', 'RankPrice'],
    key_on = 'feature.properties.LGDNAME',
    fill_color='RdYlBu',
    fill_opacity=0.9,
    line_opacity=0.2,
    highlight=True,
    
)

folium.LayerControl().add_to(m)

m.save('C:/Users/Ben/Downloads/Project/Q1/Graphs/choropleth1995.html')


# In[32]:


import pandas as pd
import os
import json

# read in population data
df = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/House Prices/UK-HPI-full-file-2018-02.csv')
df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
df = df.set_index('Date')
reduced18 = df['2018'].groupby('RegionName').mean().reset_index()
reduced18['log_avg_price'] = np.log(reduced18['AveragePrice'])
reduced18['RankPrice'] = reduced18['AveragePrice'].rank(pct=True,ascending=False)*100

from branca.utilities import split_six
quantile, scale = pd.qcut(reduced18['RankPrice'], retbins=True, q = 6 )
import folium
state_geo1 = os.path.join('data', 'C:/Users/Ben/Downloads/Project/Q1/Location/GBgeo.json')
state_geo2 = os.path.join('data', 'C:/Users/Ben/Downloads/Project/Q1/Location/IRLgeo.json')

m = folium.Map(location=[55, 4], zoom_start=5, tiles = 'cartodbpositron')
m.choropleth(
    geo_data=state_geo1,
    data=reduced18,
    columns=['RegionName', 'RankPrice'],
    key_on = 'feature.properties.LAD13NM',
    fill_color='RdYlBu',
    fill_opacity=0.9,
    line_opacity=0.2,
    legend_name='Rank of Average House Price',
    highlight=True,
    
)

m.choropleth(
    geo_data=state_geo2,
    data=reduced18,
    columns=['RegionName', 'RankPrice'],
    key_on = 'feature.properties.LGDNAME',
    fill_color='RdYlBu',
    fill_opacity=0.9,
    line_opacity=0.2,
    highlight=True,
    
)

folium.LayerControl().add_to(m)
#folium.TileLayer('CartoDB').add_to(m)
m.save('C:/Users/Ben/Downloads/Project/Q1/Graphs/choropleth2018.html')


# In[33]:


import pandas as pd
import os
import json

# read in population data
df = pd.read_csv('C:/Users/Ben/Downloads/Project/Q1/House Prices/UK-HPI-full-file-2018-02.csv')
df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
df = df.set_index('Date')
reduced05 = df['2005'].groupby('RegionName').mean().reset_index()
reduced05['log_avg_price'] = np.log(reduced05['AveragePrice'])
reduced05['RankPrice'] = reduced05['AveragePrice'].rank(pct=True,ascending=False)*100

from branca.utilities import split_six
quantile, scale = pd.qcut(reduced18['RankPrice'], retbins=True, q = 6 )
import folium
state_geo1 = os.path.join('data', 'C:/Users/Ben/Downloads/Project/Q1/Location/GBgeo.json')
state_geo2 = os.path.join('data', 'C:/Users/Ben/Downloads/Project/Q1/Location/IRLgeo.json')

m = folium.Map(location=[55, 4], zoom_start=5, tiles = 'cartodbpositron')
m.choropleth(
    geo_data=state_geo1,
    data=reduced05,
    columns=['RegionName', 'RankPrice'],
    key_on = 'feature.properties.LAD13NM',
    fill_color='RdYlBu',
    fill_opacity=0.9,
    line_opacity=0.2,
    legend_name='Rank of Average House Price',
    highlight=True,
    
)

m.choropleth(
    geo_data=state_geo2,
    data=reduced05,
    columns=['RegionName', 'RankPrice'],
    key_on = 'feature.properties.LGDNAME',
    fill_color='RdYlBu',
    fill_opacity=0.9,
    line_opacity=0.2,
    highlight=True,
    
)

folium.LayerControl().add_to(m)
#folium.TileLayer('CartoDB').add_to(m)
m.save('C:/Users/Ben/Downloads/Project/Q1/Graphs/choropleth2005.html')

