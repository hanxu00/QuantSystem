import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
from scipy import stats
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# ----------------- Market Data ------------------
#读市场组合数据，注意本地和服务器改路径
ff3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors.CSV', 
                            parse_dates={'date':[0]}, encoding='utf8', header=0)                      
ff3['date']=pd.to_datetime(ff3['date'],format = "%Y%m")
ff3['ret']=ff3['Mkt-RF']/100
ff3['logret']=np.log(1+ff3['ret'])
ff3['l2ycumlog'] = ff3['logret'].rolling(24).sum()
ff3['l2ycum']=np.exp(ff3['l2ycumlog'])-1
ff3['IB']=np.where(ff3['l2ycum']<0,1,0)
ff3['IBt-1']=ff3['IB'].shift(1)
ff3 = ff3.drop(ff3.head(25).index)
ff3['IU'] = np.where(ff3['ret']>0,1,0)
ff3[['IBt-1','IU']]=ff3[['IBt-1','IU']].astype(int)

mret = ff3[['date','ret','RF','IBt-1','IU']].reset_index()  
mret = mret.rename(columns={'ret':'mret'})
mret['monthEndDate'] = mret['date'] + MonthEnd(0)
mret = mret.drop('index',axis=1)

ff3d = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors_daily.CSV', 
                            parse_dates={'date':[0]}, encoding='utf8', header=0)               
rmrf = ff3d[['date','Mkt-RF']]
rmrf['mret'] = rmrf['Mkt-RF']/100
rmrf['vola'] = rmrf['mret'].rolling(126).std()
rmrf = rmrf.set_index('date')
rmrf = rmrf.groupby(pd.Grouper(freq='M')).last()
rmrf = rmrf.reset_index()
rmrf['monthEndDate'] = rmrf['date'] + MonthEnd(0)
rmrf = rmrf.dropna(axis = 0,subset=['vola'])
mvola = rmrf[['vola','monthEndDate']]

market = pd.merge(mret, mvola, on=['monthEndDate'], how='left')
market['state'] = np.where(market['IBt-1']==1,np.where(market['IU']==0,'Bear','Rebound'),np.where(market['IU']==0,'Correction','Bull'))

marketdat = market.groupby(['state'])['mret'].describe()[['count','mean', 'std']]
voladat = market.groupby(['state'])['vola'].describe()[['count','mean']]

# ----------------- market beta regression ------------------
portreturn11 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/portReturn_11.CSV', 
                            encoding='utf8', header=0)
portreturn11.dtypes    

#用11，1策略的数据看不同市场状态下的收益
#WMLdat = portreturn11['long_short'].describe()[['count','mean', 'std']]
WMLdat = portreturn11.groupby(['state'])['long_short'].describe()[['count','mean', 'std']]
Wdat = portreturn11.groupby(['state'])['winners'].describe()[['count','mean', 'std']]
Ldat = portreturn11.groupby(['state'])['losers'].describe()[['count','mean', 'std']]                       

WML = portreturn11[['date','long_short','mret','IBt-1','IU','vola']]
WML = WML.rename(columns={'IBt-1':'IBtm1'})
WML['volatm1'] = WML['vola'].shift(1)
WML['vartm1'] = WML['volatm1']*WML['volatm1']
WML['IBmret'] = WML['IBtm1']*WML['mret']
WML['IBIUmret'] = WML['IBtm1']*WML['IU']*WML['mret']
WML['IBvar'] = WML['IBtm1']*WML['vartm1']
#table3 market return
result = sm.formula.ols('long_short ~ 1+mret',missing='drop',data=WML).fit()
result2 = sm.formula.ols('long_short ~ 1+IBtm1+mret+IBmret',missing='drop',data=WML).fit()
result3 = sm.formula.ols('long_short ~ 1+IBtm1+mret+IBmret+IBIUmret',missing='drop',data=WML).fit()
output = summary_col([result,result2,result3],stars=True)
print(output)
#table5 lagged market variance
result4 = sm.formula.ols('long_short ~ 1+IBtm1',missing='drop',data=WML).fit()
result5 = sm.formula.ols('long_short ~ 1+vartm1',missing='drop',data=WML).fit()
result6 = sm.formula.ols('long_short ~ 1+IBtm1+vartm1',missing='drop',data=WML).fit()
result7 = sm.formula.ols('long_short ~ 1+IBvar',missing='drop',data=WML).fit()
result8 = sm.formula.ols('long_short ~ 1+IBtm1+vartm1+IBvar',missing='drop',data=WML).fit()
output2 = summary_col([result4,result5,result6,result7,result8],stars=True)
print(output2)

print(result.params)
print(result.summary())
print(result.summary())

# ----------------- 上面的基础上加入formation period回归 ------------------
for i in range(1,11):
    temp = pd.read_csv('/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/portReturn_%s.CSV' %(i), 
                            encoding='utf8', header=0)
    WML_temp = temp[['date','long_short','mret','IBt-1','IU']]
    WML_temp['formPeriod'] = i
    if i == 1:
        WML = WML_temp
    else:
        WML = pd.concat([WML,WML_temp])
WML = WML.rename(columns={'IBt-1':'IBtm1'})
WML['IBmret'] = WML['IBtm1']*WML['mret']
WML['IBIUmret'] = WML['IBtm1']*WML['IU']*WML['mret']
WML['IBfp'] = WML['IBtm1']*WML['formPeriod']
WML['IBIUfp'] = WML['IBtm1']*WML['IU']*WML['formPeriod']

result9 = sm.formula.ols('long_short ~ 1+formPeriod',missing='drop',data=WML).fit()
result10 = sm.formula.ols('long_short ~ 1+IBtm1+formPeriod+IBfp+IBIUfp+formPeriod',missing='drop',data=WML).fit()
result11 = sm.formula.ols('long_short ~ 1+IBtm1+mret+IBmret+IBIUmret+formPeriod',missing='drop',data=WML).fit()

output = summary_col([result9,result10,result11],stars=True)
print(output)

# ----------------- market risk premium regression 加入cay的预测-------------------
treasury = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/treasury bond.CSV', 
                            encoding='utf8', header=0)
treasury['date']=pd.to_datetime(treasury['caldt'],format='%Y%m%d')
treasury['monthEndDate'] = treasury['date'] + MonthEnd(0)
treasury = treasury[['b10ret','b1ret','t90ret','monthEndDate']]

AAA = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/AAA.CSV', 
                            encoding='utf8', header=0)
AAA['date']=pd.to_datetime(AAA['DATE'],format='%Y-%m-%d')
AAA['monthEndDate'] = AAA['date'] + MonthEnd(0)
AAA = AAA[['AAA','monthEndDate']]

BAA = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/BAA.CSV', 
                            encoding='utf8', header=0)
BAA['date']=pd.to_datetime(BAA['DATE'],format='%Y-%m-%d')
BAA['monthEndDate'] = BAA['date'] + MonthEnd(0)
BAA = BAA[['BAA','monthEndDate']]

cay_current = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/cay_current.CSV', 
                            encoding='utf8', header=0)                            
cay_current['date']=pd.to_datetime(cay_current['date'],format='%Y-%m-%d')
cay_current['monthEndDate'] = cay_current['date'] + MonthEnd(0)
cay_current = cay_current.rename(columns={'cay=c-0.218 a-0.801y+0.441':'cay'})
ts = pd.Series(cay_current['cay'].values, index=cay_current['monthEndDate'])
ts1 = ts.resample('M',convention='start').bfill()

temp = pd.date_range('1952-01-31', periods=2, freq='m')
s=pd.Series([0.015105,0.015105],index=temp)
ts2 = s.append(ts1, ignore_index=False, verify_integrity=True)

cay = ts2.to_frame().reset_index()
cay = cay.rename(columns={'index':'monthEndDate',0:'cay'})

merge_temp1 = pd.merge(AAA, BAA, on=['monthEndDate'], how='inner')
merge_temp2 = pd.merge(merge_temp1, treasury, on=['monthEndDate'], how='inner')
merge_temp3 = pd.merge(merge_temp2, cay, on=['monthEndDate'], how='inner')
merge_temp3['DEF'] = (merge_temp3['BAA']-merge_temp3['AAA'])/100
merge_temp3['TERM'] = merge_temp3['b10ret']-merge_temp3['b1ret']
merge_temp3 = merge_temp3.rename(columns={'t90ret':'RF'})

ff3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors.CSV', 
                            parse_dates={'date':[0]}, encoding='utf8', header=0)                      
ff3['date']=pd.to_datetime(ff3['date'],format = "%Y%m")
ff3['mret']=ff3['Mkt-RF']/100
ff3['monthEndDate'] = ff3['date'] + MonthEnd(0)
merge_temp4 = pd.merge(merge_temp3,ff3,on=['monthEndDate'], how='inner')

dataset = merge_temp4[['monthEndDate','mret','DEF','TERM','RF_x','cay']]
dataset = dataset.rename(columns={'RF_x':'RF'})
dataset['mrettp1'] = dataset['mret'].shift(-1)

dataset['coe_Int'] = np.nan
dataset['coe_DEF'] = np.nan
dataset['coe_TERM'] = np.nan
dataset['coe_RF'] = np.nan
dataset['coe_cay'] = np.nan

#用事前的数据预测expected market return
for index, row in dataset.iterrows():
    if index > 119:
        dt = dataset.iloc[0:index,:]
        result_temp = sm.formula.ols('mrettp1 ~ 1+DEF+TERM+RF+cay',missing='drop',data=dt).fit()
        dataset.loc[index+1,['coe_Int','coe_DEF','coe_TERM','coe_RF','coe_cay']] = \
            [result_temp.params['Intercept'],result_temp.params['DEF'],result_temp.params['TERM'],result_temp.params['RF'],result_temp.params['cay']]

dataset['exmret'] = dataset['coe_Int']+dataset['DEF']*dataset['coe_DEF']+dataset['TERM']*dataset['coe_TERM']\
    +dataset['RF']*dataset['coe_RF']+dataset['cay']*dataset['coe_cay']
dataset['indicator'] = np.where(dataset['exmret']*dataset['mret'] >= 0,1,0)
dataset = dataset.dropna()
#60.26%的准确率
dataset.describe()

#用整体的数据预测expected market return
result_temp = sm.formula.ols('mrettp1 ~ 1+DEF+TERM+RF+cay',missing='drop',data=dataset).fit()
dataset['coe_Int'] = result_temp.params['Intercept']
dataset['coe_DEF'] = result_temp.params['DEF']
dataset['coe_TERM'] = result_temp.params['TERM']
dataset['coe_RF'] = result_temp.params['RF']
dataset['coe_cay'] = result_temp.params['cay']
dataset['exmret'] = dataset['coe_Int']+dataset['DEF']*dataset['coe_DEF']+dataset['TERM']*dataset['coe_TERM']\
    +dataset['RF']*dataset['coe_RF']+dataset['cay']*dataset['coe_cay']
dataset['indicator'] = np.where(dataset['exmret']*dataset['mret'] >= 0,1,0)
dataset = dataset.dropna()
#62.32%的准确率
dataset['indicator'].describe()
#本月为负的时候，仅有0.14准确率，本月为正的时候有0.936864
dataset['pn'] = np.where(dataset['mret']>0,1,0)
dataset.groupby(['pn'])['indicator'].describe()[['count','mean']]

#-----------------------不加入cay的预测-----------------
treasury = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/treasury bond.CSV', 
                            encoding='utf8', header=0)
treasury['date']=pd.to_datetime(treasury['caldt'],format='%Y%m%d')
treasury['monthEndDate'] = treasury['date'] + MonthEnd(0)
treasury = treasury[['b10ret','b1ret','t90ret','monthEndDate']]

AAA = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/AAA.CSV', 
                            encoding='utf8', header=0)
AAA['date']=pd.to_datetime(AAA['DATE'],format='%Y-%m-%d')
AAA['monthEndDate'] = AAA['date'] + MonthEnd(0)
AAA = AAA[['AAA','monthEndDate']]

BAA = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/BAA.CSV', 
                            encoding='utf8', header=0)
BAA['date']=pd.to_datetime(BAA['DATE'],format='%Y-%m-%d')
BAA['monthEndDate'] = BAA['date'] + MonthEnd(0)
BAA = BAA[['BAA','monthEndDate']]

merge_temp1 = pd.merge(AAA, BAA, on=['monthEndDate'], how='inner')
merge_temp2 = pd.merge(merge_temp1, treasury, on=['monthEndDate'], how='inner')
merge_temp2['DEF'] = (merge_temp2['BAA']-merge_temp2['AAA'])/100
merge_temp2['TERM'] = merge_temp2['b10ret']-merge_temp2['b1ret']
merge_temp2 = merge_temp2.rename(columns={'t90ret':'RF'})

ff3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors.CSV', 
                            parse_dates={'date':[0]}, encoding='utf8', header=0)                      
ff3['date']=pd.to_datetime(ff3['date'],format = "%Y%m")
ff3['mret']=ff3['Mkt-RF']/100
ff3['monthEndDate'] = ff3['date'] + MonthEnd(0)
merge_temp3 = pd.merge(merge_temp2,ff3,on=['monthEndDate'], how='inner')

dataset = merge_temp3[['monthEndDate','mret','DEF','TERM','RF_x']]
dataset = dataset.rename(columns={'RF_x':'RF'})
dataset['mrettp1'] = dataset['mret'].shift(-1)

#用整体的数据预测expected market return
result_temp = sm.formula.ols('mrettp1 ~ 1+DEF+TERM+RF',missing='drop',data=dataset).fit()
dataset['coe_Int'] = result_temp.params['Intercept']
dataset['coe_DEF'] = result_temp.params['DEF']
dataset['coe_TERM'] = result_temp.params['TERM']
dataset['coe_RF'] = result_temp.params['RF']
dataset['exmret'] = dataset['coe_Int']+dataset['DEF']*dataset['coe_DEF']+dataset['TERM']*dataset['coe_TERM']\
    +dataset['RF']*dataset['coe_RF']
dataset['indicator'] = np.where(dataset['exmret']*dataset['mret'] >= 0,1,0)
dataset = dataset.dropna()
#63.63%的准确率
dataset['indicator'].describe()
#本月为负的时候，仅有0.12准确率，本月为正的时候有0.96
dataset['pn'] = np.where(dataset['mret']>0,1,0)
dataset.groupby(['pn'])['indicator'].describe()[['count','mean']]

#用事前的数据预测expected market return
dataset['coe_Int'] = np.nan
dataset['coe_DEF'] = np.nan
dataset['coe_TERM'] = np.nan
dataset['coe_RF'] = np.nan

for index, row in dataset.iterrows():
    if index > 119:
        dt = dataset.iloc[0:index,:]
        result_temp = sm.formula.ols('mrettp1 ~ 1+DEF+TERM+RF',missing='drop',data=dt).fit()
        dataset.loc[index+1,['coe_Int','coe_DEF','coe_TERM','coe_RF']] = \
            [result_temp.params['Intercept'],result_temp.params['DEF'],result_temp.params['TERM'],result_temp.params['RF']]

dataset['exmret'] = dataset['coe_Int']+dataset['DEF']*dataset['coe_DEF']+dataset['TERM']*dataset['coe_TERM']\
    +dataset['RF']*dataset['coe_RF']
dataset['indicator'] = np.where(dataset['exmret']*dataset['mret'] >= 0,1,0)
dataset = dataset.dropna()
#62.31%的准确率
dataset['indicator'].describe()
#本月为负的时候，仅有0.15准确率，本月为正的时候有0.92
dataset['pn'] = np.where(dataset['mret']>0,1,0)
dataset.groupby(['pn'])['indicator'].describe()[['count','mean']]

EMRP = dataset[['monthEndDate','indicator']]
EMRP.to_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/EMRP.CSV')

#--------------------drnamic和非dynamic相同时间段内的收益对比---------------------
ewretdat3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/portReturn_11.CSV', 
                            encoding='utf8', header=0)
ewretdat3.dtypes
ewretdat3['DATE']=pd.to_datetime(ewretdat3['date'],format="%Y-%m-%d")
ewretdat3 = ewretdat3[ewretdat3['DATE']>'1941-06-30']

mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
mom_mean = mom_mean.reset_index().rename(columns={0:'mean','index':'momr'})
mom_mean

t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T

t_losers['momr']='losers'
t_winners['momr']='winners'
t_long_short['momr']='long_short'

t_output =pd.concat([t_winners, t_losers, t_long_short])\
    .rename(columns={0:'t-stat', 1:'p-value'})

mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
print(mom_output)
