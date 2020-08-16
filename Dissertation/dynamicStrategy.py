import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
from scipy import stats

#股票数据获取
conn=wrds.Connection(wrds_username='hanxu00')
crsp = conn.raw_sql("""
                    select a.permno, a.permco, b.ncusip, a.date, 
                    b.shrcd, b.exchcd, b.siccd,
                    a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                    from crsp.msf as a
                    left join crsp.msenames as b
                    on a.permno=b.permno
                    and b.namedt<=a.date
                    and a.date<=b.nameendt
                    where a.date between '07/01/1941' and '12/31/2019'
                    and b.exchcd between -2 and 2
                    and b.shrcd between 10 and 11
                    """) 
conn.close()
crsp[['permco','permno','shrcd','exchcd']]=\
    crsp[['permco','permno','shrcd','exchcd']].astype(int)
crsp['date']=pd.to_datetime(crsp['date'])
crsp['monthEndDate'] = crsp['date'] + MonthEnd(0)
print('crsp数据集的行列数：\n',crsp.shape)

# -----------------读市场组合数据，注意本地和服务器改路径------------------
#ff3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors.CSV', 
                            #parse_dates={'date':[0]}, encoding='utf8', header=0)
ff3 = pd.read_csv('F-F_Research_Data_Factors.CSV',parse_dates={'date':[0]}, encoding='utf8', header=0)                      
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

#EMRP = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/EMRP.CSV', 
                            #encoding='utf8', header=0)
EMRP = pd.read_csv('EMRP.CSV',encoding='utf8', header=0)
EMRP['date']=pd.to_datetime(EMRP['monthEndDate'],format='%Y-%m-%d')
EMRP['monthEndDate'] = EMRP['date'] + MonthEnd(0)
EMRP = EMRP[['monthEndDate','indicator']]

market = pd.merge(mret, EMRP, on=['monthEndDate'], how='inner')
market['eIU'] = market['indicator'].shift(1)
market = market.dropna()
market['eIU'] = market['eIU'].astype(int)
market['estate'] = np.where(market['IBt-1']==1,np.where(market['eIU']==0,'Bear','Rebound'),np.where(market['eIU']==0,'Correction','Bull'))
market = market.drop(['date'],axis=1)
#marketdat = market.groupby(['state'])['mret'].describe()[['count','mean', 'std']]
#voladat = market.groupby(['state'])['vola'].describe()[['count','mean']]
print('market数据集的行列数：\n',market.shape)

crsp_market = pd.merge(crsp, market, on=['monthEndDate'], how='left')
umd = crsp_market[['permno','date','monthEndDate','ret','estate']].sort_values(['permno','date'])
umd['ret']=umd['ret'].fillna(0)
umd['logret']=np.log(1+umd['ret'])
umd = umd.reset_index()
umd = umd.drop(['index'], axis = 1)
umd['cumlret'] = np.nan

#-----------最重要的循环！计算formation period里的累计收益------------
for i in range(12, len(umd)-1):
    if umd.loc[i]['estate'] == 'Rebound':
        if umd.loc[i]['permno'] == umd.loc[i-2]['permno']: 
                umd.loc[i,'cumlret'] = umd.loc[i-2,'logret']
    else:
        if umd.loc[i]['permno'] == umd.loc[i-12]['permno']: 
                umd.loc[i,'cumlret'] = umd.loc[i-12:i-2,'logret'].sum()

umd['cumret']=np.exp(umd['cumlret'])-1
#防止有些cumret为NAN，之前是删除NAN，现在直接删一年的来保证连续
umd = umd.drop(umd.groupby(['permno']).head(12).index,axis=0)

umd['momr']=umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))
umd=umd.dropna(axis=0, subset=['momr'])
umd.momr=umd.momr.astype(int)
umd['momr'] = umd['momr']+1
#umd2 = umd.groupby(['date','momr'])['ret'].mean().reset_index()

#------------------组合计算全部完毕，下面开始统计平均收益-------------------
ewret = umd.groupby(['date','momr'])['ret'].mean().reset_index()
ewretdat = ewret.sort_values(by=['momr'])
# portfolio summary
ewretdat.groupby(['momr'])['ret'].describe()[['count','mean', 'std']]

ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ret')      
ewretdat2 = ewretdat2.add_prefix('port')
ewretdat2 = ewretdat2.rename(columns={'port1':'losers', 'port10':'winners'})
ewretdat2['long_short'] = ewretdat2['winners'] - ewretdat2['losers']
        
ewretdat3 = ewretdat2
ewretdat3['1+losers']=1+ewretdat3['losers']
ewretdat3['1+winners']=1+ewretdat3['winners']
ewretdat3['1+ls'] = 1+ewretdat3['long_short']
ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1
#这里可以输出sample时间段内的winner，loser和WML的累计收益率
ewretdat3.to_csv('cumret_strategy.csv')
#ewretdat3.to_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/dynamicEwretdat3.csv')

mom_mean = ewretdat3[['winners', 'losers', 'long_short', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
#mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
mom_mean = mom_mean.rename(columns={0:'mean'}).reset_index()

#-------------上面平均收益率统计完毕，下面是统计t和p值-----------------
t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T
t_2 = pd.Series(stats.ttest_1samp(ewretdat3['port2'],0.0)).to_frame().T
t_3 = pd.Series(stats.ttest_1samp(ewretdat3['port3'],0.0)).to_frame().T
t_4 = pd.Series(stats.ttest_1samp(ewretdat3['port4'],0.0)).to_frame().T
t_5 = pd.Series(stats.ttest_1samp(ewretdat3['port5'],0.0)).to_frame().T
t_6 = pd.Series(stats.ttest_1samp(ewretdat3['port6'],0.0)).to_frame().T
t_7 = pd.Series(stats.ttest_1samp(ewretdat3['port7'],0.0)).to_frame().T
t_8 = pd.Series(stats.ttest_1samp(ewretdat3['port8'],0.0)).to_frame().T
t_9 = pd.Series(stats.ttest_1samp(ewretdat3['port9'],0.0)).to_frame().T

t_losers['momr']='losers'
t_winners['momr']='winners'
t_long_short['momr']='long_short'
t_2['momr']='port2'
t_3['momr']='port3'
t_4['momr']='port4'
t_5['momr']='port5'
t_6['momr']='port6'
t_7['momr']='port7'
t_8['momr']='port8'
t_9['momr']='port9'

t_output =pd.concat([t_winners, t_losers, t_long_short, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9])\
     .rename(columns={0:'t-stat', 1:'p-value'})
mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
mom_output.to_csv('dynamic_output.csv')

