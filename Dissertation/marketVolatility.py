# 7.18 加入可以从服务器上运行的路径
# 7.19 算volatility，formation的时候skip 1 month，
#      在两处体现，一是在找市场最大波动时，二是在主循环中。
# 7.20 稍作修改。格式化代码，删除无用部分。

import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
from scipy import stats

#---------------拉股票月度数据，除了改日期外基本不动------------------
conn=wrds.Connection(wrds_username='hanxu00')
crsp_m = conn.raw_sql("""
                            select a.permno, a.permco, b.ncusip, a.date, 
                            b.shrcd, b.exchcd, b.siccd,
                            a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                            from crsp.msf as a
                            left join crsp.msenames as b
                            on a.permno=b.permno
                            and b.namedt<=a.date
                            and a.date<=b.nameendt
                            where a.date between '01/01/1991' and '6/30/2020'
                            and b.exchcd between -2 and 2
                            and b.shrcd between 10 and 11
                            """) 
conn.close()
crsp_m[['permco','permno','shrcd','exchcd']]=\
    crsp_m[['permco','permno','shrcd','exchcd']].astype(int)
crsp_m['date']=pd.to_datetime(crsp_m['date'])
crsp_m['monthEndDate'] = crsp_m['date'] + MonthEnd(0)

# -----------------读市场组合数据，注意本地和服务器改路径------------------
# ff3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors_daily.CSV', 
#                             skiprows=3, skipfooter=1, parse_dates={'date':[0]}, encoding='utf8')               
ff3 = pd.read_csv('F-F_Research_Data_Factors_daily.CSV',skiprows=3, skipfooter=1, parse_dates={'date':[0]}, encoding='utf8')                       

#-----------计算市场组合t-12到t-1月份累计波动最大的月份--------------------
rmrf = ff3[['date','Mkt-RF']]
#skip 1 month,所以只循环到11，然后再shift往上移一下
formationVola = [1,2,3,4,5,6,7,8,9,10,11]
for i in formationVola:
    rmrf[str(i)] = rmrf['Mkt-RF'].rolling(21*i).std()
rmrf = rmrf.set_index('date')
rmrf = rmrf.groupby(pd.Grouper(freq='M')).last()

rmrfmv = rmrf.drop(['Mkt-RF'], axis = 1)
rmrfmv = rmrfmv.dropna(axis = 0, subset=['11'])
# rmrfmv['maxVolaMon'] = rmrfmv.idxmax(axis=1)
# rmrfmv['maxVola'] = rmrfmv.max(axis=1)
rmrfmv['maxVolaMon'] = rmrfmv.idxmin(axis=1)
rmrfmv['maxVola'] = rmrfmv.(axis=1)
rmrfmv['maxVolaMon'] = rmrfmv['maxVolaMon'].astype('int')

rmrfMaxVola = rmrfmv[['maxVolaMon','maxVola']]
rmrfMaxVola = rmrfMaxVola.reset_index()
rmrfMaxVola['monthEndDate'] = rmrfMaxVola['date'] + MonthEnd(0)
rmrfMaxVola['maxVolaMon_skip1'] = rmrfMaxVola['maxVolaMon'].shift(1)
rmrfMaxVola['maxVola_skip1'] = rmrfMaxVola['maxVola'].shift(1)

crsp_mv = pd.merge(crsp_m, rmrfMaxVola, on=['monthEndDate'], how='left')

umd = crsp_mv[['permno','date_x','monthEndDate','ret','maxVolaMon_skip1']].sort_values(['permno','date_x'])
umd['ret']=umd['ret'].fillna(0)
umd['logret']=np.log(1+umd['ret'])
umd = umd.reset_index()
umd = umd.drop(['index'], axis = 1)

#-----------最重要的循环！计算formation period里的累计收益------------
for i in range(1, len(umd)-1):
    maxV = umd.loc[i]['maxVolaMon_skip1']
    if np.isnan(maxV) == False:
        if i-maxV-1 >= 0:
            if umd.loc[i]['permno'] == umd.loc[i-maxV-1]['permno']:
                umd.loc[i,'cumret'] = umd.loc[i-maxV-1:i-1,'logret'].sum()
umd['cumret']=np.exp(umd['cumret'])-1
#防止有些cumret为NAN，之前是删除NAN，现在直接删一年的来保证连续
umd = umd.drop(umd.groupby(['permno']).head(12).index,axis=0)
#umd=umd.dropna(axis=0, subset=['cumret'])
umd['momr']=umd.groupby('date_x')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))
umd=umd.dropna(axis=0, subset=['momr'])
umd.momr=umd.momr.astype(int)
umd['momr'] = umd['momr']+1
umd['holdEnd'] = umd['date_x']+MonthEnd(1)
# 这里因为只持有一个月，就直接把下个月的收益shift过来了。如果持有多个月的话，还是按原来代码里的那种merge方法来。
umd['holdRet'] = umd.groupby('permno')['ret'].shift(-1)
umd = umd.dropna(axis=0, subset=['holdRet'])

# umd.to_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/dynamicUmd.csv')
umd2 = umd.sort_values(by=['date_x','momr','permno']).drop_duplicates()
umd3 = umd2.groupby(['date_x','momr'])['holdRet'].mean().reset_index()

#------------------组合计算全部完毕，下面开始统计平均收益-------------------
ewret = umd3.groupby(['date_x','momr'])['holdRet'].mean().reset_index()
ewretdat = ewret.sort_values(by=['momr'])
# portfolio summary
ewretdat.groupby(['momr'])['holdRet'].describe()[['count','mean', 'std']]

ewretdat2 = ewretdat.pivot(index='date_x', columns='momr', values='holdRet')      
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
ewretdat3.to_csv('cumret_strategy')
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
mom_output.to_csv('mom_output.csv')
