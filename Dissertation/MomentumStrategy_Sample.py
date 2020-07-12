################################################
# Jegadeesh & Titman (1993) Momentum Portfolio #
# June 2019                                    #  
# Qingyi (Freda) Song Drechsler                #
################################################

import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats

###################
# Connect to WRDS #
###################
conn=wrds.Connection(wrds_username='hanxu00')

###################
# CRSP Block      #
###################
# sql similar to crspmerge macro
# added exchcd=-2,-1,0 to address the issue that stocks temp stopped trading
# without exchcd=-2,-1, 0 the non-trading months will be tossed out in the output
# leading to wrong cumret calculation in momentum step
# Code	Definition
# -2	Halted by the NYSE or AMEX
# -1	Suspended by the NYSE, AMEX, or NASDAQ
# 0	Not Trading on NYSE, AMEX, or NASDAQ
# 1	New York Stock Exchange
# 2	American Stock Exchange


# 表及字段含义：https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_m_stock/
# crsp.msf=Monthly Stock - Securities 
# crsp.msenames= CRSP, Monthly Stock Event - Name History
# permno股票识别码, permco公司识别码, ncusip：股票The CUSIP Agency will often change an issue's CUSIP identifier to reflect a change of name or capital structure.
# shrcd=share code, exchcd=Exchange Code, siccd=industry, 
# ret=return, vol=volume, shrout=Shares Outstanding, prc=Price or Bid/Ask Average
# cfacpr=Cumulative Factor to Adjust Prices
# cfacshr=Cumulative Factor to Adjust Shares/Vol
# namedt=Names Date, nameendt=Names Ending Date
# exchcd=Exchange Code
# shrcd 第一位1=Ordinary Common Shares，
# 第二位0=Securities which have not been further defined.
# 1=Securities which need not be further defined.

conn.list_tables(library="crsp")

crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, b.ncusip, a.date, 
                      b.shrcd, b.exchcd, b.siccd,
                      a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1963' and '12/31/1989'
                      and b.exchcd between -2 and 2
                      and b.shrcd between 10 and 11
                      """) 
crsp_m.head()
print('数据集的行列数：\n',crsp_m.shape)
# Change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=\
    crsp_m[['permco','permno','shrcd','exchcd']].astype(int)
crsp_m.dtypes
# Line up date to be end of month
crsp_m['date']=pd.to_datetime(crsp_m['date'])
crsp_m.dtypes
#######################################################
# Create Momentum Portfolio                           #   
# Measures Based on Past (J) Month Compounded Returns #
#######################################################

J = 6 # Formation Period Length: J can be between 3 to 12 months
K = 6 # Holding Period Length: K can be between 3 to 12 months

#sort_values作用: pandas模块，Sort by the values along either axis.
_tmp_crsp = crsp_m[['permno','date','ret']].sort_values(['permno','date'])\
    .set_index('date')
_tmp_crsp.head()
# Replace missing return with 0, fillna作用: Fill NA/NaN values using the specified method.
print('数据集中是否存在缺失值：\n',any(_tmp_crsp['ret'].isnull()))
_tmp_crsp['ret']=_tmp_crsp['ret'].fillna(0)
print('数据集中是否存在缺失值：\n',any(_tmp_crsp['ret'].isnull()))

# Calculate rolling cumulative return
# by summing log(1+ret) over the formation period
_tmp_crsp['logret']=np.log(1+_tmp_crsp['ret'])
#groupby是pandas的函数，可以类似stata里的by用，后面可以直接.function
#rolling(window,min_periods,center)
#window: size of moving window
#min_periods: threshold of non-null data points to require (otherwise result is NA)
#center: boolean, whether to set the labels at the center (default is False)
umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(J, min_periods=J).sum()
#reset_index行索引设置为变量
umd = umd.reset_index()
umd['cumret']=np.exp(umd['logret'])-1
umd.head()

########################################
# Formation of 10 Momentum Portfolios  #
########################################

# For each date: assign ranking 1-10 based on cumret
# 1=lowest 10=highest cumret
umd=umd.dropna(axis=0, subset=['cumret'])

#transform函数后面再学一下
umd['momr']=umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))

umd.momr=umd.momr.astype(int)
umd['momr'] = umd['momr']+1
umd.head()

# Corrected previous version month end line up issue
# First lineup date to month end date medate
# Then calculate hdate1 and hdate2 using medate

# MonthEnd MonthBegin是日期便偏移函数，详细的看一下pandas时间序列相关内容
# https://www.pypandas.cn/docs/user_guide/timeseries.html#日期-时间组件
umd['form_date'] = umd['date']
umd['medate'] = umd['date']+MonthEnd(0)
umd['hdate1']=umd['medate']+MonthBegin(1)
umd['hdate2']=umd['medate']+MonthEnd(K)

umd = umd[['permno', 'form_date','momr','hdate1','hdate2']]
umd.head()
# join rank and return data together
# note: this step consumes a lot of memory so takes a while
_tmp_ret = crsp_m[['permno','date','ret']]
port = pd.merge(_tmp_ret, umd, on=['permno'], how='inner')
# merge是按permno来的，所以_tmp_ret里每个permno和date后面都跟了umd里每个permno里的数据，按下面这个规则可以剔除
port = port[(port['hdate1']<=port['date']) & (port['date']<=port['hdate2'])]

umd2 = port.sort_values(by=['date','momr','form_date','permno']).drop_duplicates()
umd3 = umd2.groupby(['date','momr','form_date'])['ret'].mean().reset_index()

umd2.head()
umd2.shape
umd3.head()
umd3.shape

# Skip first two years of the sample 
start_yr = umd3['date'].dt.year.min()+2
umd3 = umd3[umd3['date'].dt.year>=start_yr]
umd3 = umd3.sort_values(by=['date','momr'])

# Create one return series per MOM group every month
ewret = umd3.groupby(['date','momr'])['ret'].mean().reset_index()
ewstd = umd3.groupby(['date','momr'])['ret'].std().reset_index()
ewret = ewret.rename(columns={'ret':'ewret'})
ewstd = ewstd.rename(columns={'ret':'ewretstd'})
ewret.head()
ewret.shape
ewstd.head()
ewstd.shape
ewretdat = pd.merge(ewret, ewstd, on=['date','momr'], how='inner')
ewretdat = ewretdat.sort_values(by=['momr'])
ewretdat.head()
ewretdat.shape
# portfolio summary
ewretdat.groupby(['momr'])['ewret'].describe()[['count','mean', 'std']]

#################################
# Long-Short Portfolio Returns  #
#################################

# Transpose portfolio layout to have columns as portfolio returns
# pivot这个好用！:Return reshaped DataFrame organized by given index / column values.
ewretdat2 = ewretdat.pivot(index='date', columns='momr', values='ewret')
ewretdat2.head()
# Add prefix port in front of each column. prefix:批量重命名
ewretdat2 = ewretdat2.add_prefix('port')
ewretdat2 = ewretdat2.rename(columns={'port1':'losers', 'port10':'winners'})
ewretdat2['long_short'] = ewretdat2['winners'] - ewretdat2['losers']
ewretdat2.head()
# Compute Long-Short Portfolio Cumulative Returns
ewretdat3 = ewretdat2
ewretdat3['1+losers']=1+ewretdat3['losers']
ewretdat3['1+winners']=1+ewretdat3['winners']
ewretdat3['1+ls'] = 1+ewretdat3['long_short']
#cumprod一个连乘函数，和prod有啥区别？
ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1
ewretdat3.head()
#################################
# Portfolio Summary Statistics  #
################################# 

# Mean 
mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
mom_mean = mom_mean.rename(columns={0:'mean'}).reset_index()
mom_mean
# T-Value and P-Value
# DataFrame.T: Transpose index and columns.
t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T

t_losers['momr']='losers'
t_winners['momr']='winners'
t_long_short['momr']='long_short'

t_output =pd.concat([t_winners, t_losers, t_long_short])\
    .rename(columns={0:'t-stat', 1:'p-value'})
t_output
# Combine mean, t and p
mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
mom_output
conn.close()