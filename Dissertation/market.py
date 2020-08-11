import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
from scipy import stats

# -----------------读市场组合数据，注意本地和服务器改路径------------------
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