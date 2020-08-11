import pandas as pd
import numpy as np
import datetime as dt
import wrds
import psycopg2 
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
import statsmodels.api as sm

conn=wrds.Connection(wrds_username='hanxu00')

###################
# Compustat Block #
###################
#变量说明
#indfmt: INDL除金融外, FS金融; popsrc='D' Domestic;	consol='C' Consolidated
#Fama
#gvkey=Global Company Key; at=Assets - Total; pstkl=Preferred Stock - Liquidating Value
#txditc=Deferred Taxes and Investment Tax Credit; pstkrv=Preferred Stock - Redemption Value
#seq=Stockholders Equity - Parent;pstk=Preferred/Preference Stock (Capital) - Total
#根据文章另找的
#oiadp=Operating Income After Depreciation; dvt=Dividends - Total;
comp = conn.raw_sql("""
                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk, oiadp, dvt
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D' 
                    and consol='C'
                    and datadate between '01/01/1990' and '12/31/1999'
                    """)

comp['datadate']=pd.to_datetime(comp['datadate'])
comp['year']=comp['datadate'].dt.year

# create preferrerd stock
comp['ps']=np.where(comp['pstkrv'].isnull(), comp['pstkl'], comp['pstkrv'])
comp['ps']=np.where(comp['ps'].isnull(),comp['pstk'], comp['ps'])
comp['ps']=np.where(comp['ps'].isnull(),0,comp['ps'])

comp['txditc']=comp['txditc'].fillna(0)

# create book equity
comp['be']=comp['seq']+comp['txditc']-comp['ps']
comp['be']=np.where(comp['be']>0, comp['be'], np.nan)

# number of years in Compustat
comp=comp.sort_values(by=['gvkey','datadate'])
comp['count']=comp.groupby(['gvkey']).cumcount()

comp=comp[['gvkey','datadate','year','be','count','oiadp','dvt','at']]

###################
# CRSP Block      #
###################
# -2 Halted by the NYSE or AMEX
# -1 Suspended by the NYSE, AMEX, or NASDAQ
# 0	Not Trading on NYSE, AMEX, or NASDAQ
# 1	New York Stock Exchange
# 2	American Stock Exchange
# retx = return without dividend
crsp_m = conn.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1990' and '12/31/1999'
                      and b.exchcd between 1 and 3
                      """) 

# change variable format to int
crsp_m[['permco','permno','shrcd','exchcd']]=crsp_m[['permco','permno','shrcd','exchcd']].astype(int)

# Line up date to be end of month
crsp_m['date']=pd.to_datetime(crsp_m['date'])
crsp_m['jdate']=crsp_m['date']+MonthEnd(0)

# add delisting return
dlret = conn.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """)
dlret.permno=dlret.permno.astype(int)
dlret['dlstdt']=pd.to_datetime(dlret['dlstdt'])
dlret['jdate']=dlret['dlstdt']+MonthEnd(0)

crsp = pd.merge(crsp_m, dlret, how='left',on=['permno','jdate'])
crsp['dlret']=crsp['dlret'].fillna(0)
crsp['ret']=crsp['ret'].fillna(0)
crsp['retadj']=(1+crsp['ret'])*(1+crsp['dlret'])-1
crsp['me']=crsp['prc'].abs()*crsp['shrout'] # calculate market equity
crsp=crsp.drop(['dlret','dlstdt','prc','shrout'], axis=1)
crsp=crsp.sort_values(by=['jdate','permco','me'])

### Aggregate Market Cap ###
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate','permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['jdate','permco'])['me'].max().reset_index()
# join by jdate/maxme to find the permno
crsp1=pd.merge(crsp, crsp_maxme, how='inner', on=['jdate','permco','me'])
# drop me column and replace with the sum me
crsp1=crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['jdate','permco'])
# sort by permno and date and also drop duplicates
crsp2=crsp2.sort_values(by=['permno','jdate']).drop_duplicates()

# keep December market cap
crsp2['year']=crsp2['jdate'].dt.year
crsp2['month']=crsp2['jdate'].dt.month
decme=crsp2[crsp2['month']==12]
decme=decme[['permno','date','jdate','me','year']].rename(columns={'me':'dec_me'})

### July to June dates
crsp2['ffdate']=crsp2['jdate']+MonthEnd(-6)
crsp2['ffyear']=crsp2['ffdate'].dt.year
crsp2['ffmonth']=crsp2['ffdate'].dt.month
crsp2['1+retx']=1+crsp2['retx']
crsp2=crsp2.sort_values(by=['permno','date'])

# 每只股票按每财年算累计收益
crsp2['cumretx']=crsp2.groupby(['permno','ffyear'])['1+retx'].cumprod()
# lag cumret，本财年截止到上一个月的累计收益
crsp2['lcumretx']=crsp2.groupby(['permno'])['cumretx'].shift(1)

# lag market cap
crsp2['lme']=crsp2.groupby(['permno'])['me'].shift(1)

# if first permno then use me/(1+retx) to replace the missing value
crsp2['count']=crsp2.groupby(['permno']).cumcount()
crsp2['lme']=np.where(crsp2['count']==0, crsp2['me']/crsp2['1+retx'], crsp2['lme'])

# baseline me，也就是上一财年末的me
mebase=crsp2[crsp2['ffmonth']==1][['permno','ffyear', 'lme']].rename(columns={'lme':'mebase'})

# merge result back together，crsp3包含了企业的月度数据
crsp3=pd.merge(crsp2, mebase, how='left', on=['permno','ffyear'])
crsp3['wt']=np.where(crsp3['ffmonth']==1, crsp3['lme'], crsp3['mebase']*crsp3['lcumretx'])

decme['year']=decme['year']+1
decme=decme[['permno','year','dec_me']]

# Info as of June
crsp3_jun = crsp3[crsp3['month']==6]
# crsp_jun包含了企业的财年度数据
crsp_jun = pd.merge(crsp3_jun, decme, how='inner', on=['permno','year'])
crsp_jun=crsp_jun[['permno','date', 'jdate', 'shrcd','exchcd','retadj','me','wt','cumretx','mebase','lme','dec_me']]
crsp_jun=crsp_jun.sort_values(by=['permno','jdate']).drop_duplicates()

#######################
# CCM Block           #
#######################
ccm=conn.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)
conn.close()
ccm['linkdt']=pd.to_datetime(ccm['linkdt'])
ccm['linkenddt']=pd.to_datetime(ccm['linkenddt'])
# if linkenddt is missing then set to today date
ccm['linkenddt']=ccm['linkenddt'].fillna(pd.to_datetime('today'))

#ccm1连接compustat和ccm
ccm1=pd.merge(comp[['gvkey','datadate','be','count','oiadp','dvt','at']],ccm,how='left',on=['gvkey'])
ccm1['yearend']=ccm1['datadate']+YearEnd(0)
ccm1['jdate']=ccm1['yearend']+MonthEnd(6)

# set link date bounds，CCM连接日期的使用规则
ccm2=ccm1[(ccm1['jdate']>=ccm1['linkdt'])&(ccm1['jdate']<=ccm1['linkenddt'])]
ccm2=ccm2[['gvkey','permno','datadate','yearend', 'jdate','be', 'count','oiadp','dvt','at']]

# link comp and crsp，ccm_jun，合并后财年度数据
ccm_jun=pd.merge(crsp_jun, ccm2, how='inner', on=['permno', 'jdate'])
ccm_jun['beme']=ccm_jun['be']*1000/ccm_jun['dec_me']

#######################
# NYSE breakpoints    #
#######################
# select NYSE stocks for bucket breakdown
# exchcd = 1 and positive beme and positive me and shrcd in (10,11) and at least 2 years in comp
nyse=ccm_jun[(ccm_jun['exchcd']==1) & (ccm_jun['beme']>0) & (ccm_jun['me']>0) & (ccm_jun['count']>1) & ((ccm_jun['shrcd']==10) | (ccm_jun['shrcd']==11))]
# size分十组
#nyse_sz=nyse.groupby(['jdate'])['me'].median().to_frame().reset_index().rename(columns={'me':'sizemedn'})
nyse_sz=nyse.groupby(['jdate'])['me'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()
nyse_sz=nyse_sz[['jdate','10%','20%','30%','40%','50%','60%','70%','80%','90%']].rename(columns={'10%':'sz10', '20%':'sz20', '30%':'sz30', '40%':'sz40', '50%':'sz50', '60%':'sz60', '70%':'sz70', '80%':'sz80', '90%':'sz90'})
# beme breakdown暂时不用bm的分组
# nyse_bm=nyse.groupby(['jdate'])['beme'].describe(percentiles=[0.3, 0.7]).reset_index()
# nyse_bm=nyse_bm[['jdate','30%','70%']].rename(columns={'30%':'bm30', '70%':'bm70'})
# nyse_breaks = pd.merge(nyse_sz, nyse_bm, how='inner', on=['jdate'])

# join back size and beme breakdown
ccm1_jun = pd.merge(ccm_jun, nyse_sz, how='left', on=['jdate'])

# function to assign sz and bm bucket
def sz_bucket(row):
    if row['me']==np.nan:
        value=''
    elif row['me']<=row['sz10']:
        value='1'
    elif row['me']<=row['sz20']:
        value='2'
    elif row['me']<=row['sz30']:
        value='3'
    elif row['me']<=row['sz40']:
        value='4'
    elif row['me']<=row['sz50']:
        value='5'
    elif row['me']<=row['sz60']:
        value='6'
    elif row['me']<=row['sz70']:
        value='7'  
    elif row['me']<=row['sz80']:
        value='8'  
    elif row['me']<=row['sz90']:
        value='9'                                                         
    elif row['me']>row['sz90']:
        value='10'
    else:
        value=''
    return value

# def bm_bucket(row):
#     if 0<=row['beme']<=row['bm30']:
#         value = 'L'
#     elif row['beme']<=row['bm70']:
#         value='M'
#     elif row['beme']>row['bm70']:
#         value='H'
#     else:
#         value=''
#     return value

# assign size portfolio
ccm1_jun['szport']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(sz_bucket, axis=1), '')
# assign book-to-market portfolio
#ccm1_jun['bmport']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), ccm1_jun.apply(bm_bucket, axis=1), '')
# create positivebmeme and nonmissport variable
ccm1_jun['posbm']=np.where((ccm1_jun['beme']>0)&(ccm1_jun['me']>0)&(ccm1_jun['count']>=1), 1, 0)
ccm1_jun['nonmissport']=np.where((ccm1_jun['szport']!=''), 1, 0)

# store portfolio assignment as of June
june=ccm1_jun[['permno','date', 'jdate','szport','posbm','nonmissport']]
june['ffyear']=june['jdate'].dt.year

# merge back with monthly records
crsp3 = crsp3[['date','permno','shrcd','exchcd','retadj','me','wt','cumretx','ffyear','jdate']]
ccm3=pd.merge(crsp3, 
        june[['permno','ffyear','szport','posbm','nonmissport']], how='left', on=['permno','ffyear'])

# keeping only records that meet the criteria,ccm4是合并了size分组后的公司收益月度数据
ccm4=ccm3[(ccm3['wt']>0)& (ccm3['posbm']==1) & (ccm3['nonmissport']==1) & 
          ((ccm3['shrcd']==10) | (ccm3['shrcd']==11))]

#把无风险利率合入ccm4
ff3 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/SizeEffect/F-F_Research_Data_Factors.CSV', 
                            parse_dates={'date':[0]}, encoding='utf8', header=0)               
rf = ff3[['date','RF']]
rf['date']=pd.to_datetime(rf['date'],format = "%Y%m")
rf['jdate'] = rf['date'] + MonthEnd(0)
rf=rf[['jdate','RF']]
ccm5 =pd.merge(ccm4, rf, how='left', on=['jdate'])
ccm5['eret']=ccm5['retadj']-ccm5['RF']/100
 
############################
# table 1                  #
############################
# 各分组公司计数
def conutfirm(data):
    num_firm=data.groupby(['jdate','szport'])['retadj'].count().reset_index().rename(columns={'retadj':'n_firms'})
    num_firmdat = num_firm.sort_values(by=['szport'])
    num_firmdat = num_firmdat.pivot(index='jdate', columns='szport', values='n_firms')      
    num_firmdat = num_firmdat.add_prefix('port')
    num_firmdat = num_firmdat.rename(columns={'port1':'S', 'port10':'B'})
    num_firmdat_mean = num_firmdat[['S', 'B', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
    num_firmdat_mean = num_firmdat_mean.rename(columns={0:'mean'}).reset_index()
    return num_firmdat_mean

#算t值
def sz_t(group):
    t_S = pd.Series(stats.ttest_1samp(group['S'],0.0)).to_frame().T
    t_B = pd.Series(stats.ttest_1samp(group['B'],0.0)).to_frame().T
    t_SMB = pd.Series(stats.ttest_1samp(group['SMB'],0.0)).to_frame().T
    t_2 = pd.Series(stats.ttest_1samp(group['port2'],0.0)).to_frame().T
    t_3 = pd.Series(stats.ttest_1samp(group['port3'],0.0)).to_frame().T
    t_4 = pd.Series(stats.ttest_1samp(group['port4'],0.0)).to_frame().T
    t_5 = pd.Series(stats.ttest_1samp(group['port5'],0.0)).to_frame().T
    t_6 = pd.Series(stats.ttest_1samp(group['port6'],0.0)).to_frame().T
    t_7 = pd.Series(stats.ttest_1samp(group['port7'],0.0)).to_frame().T
    t_8 = pd.Series(stats.ttest_1samp(group['port8'],0.0)).to_frame().T
    t_9 = pd.Series(stats.ttest_1samp(group['port9'],0.0)).to_frame().T

    t_S['szport']='S'
    t_B['szport']='B'
    t_SMB['szport']='SMB'
    t_2['szport']='port2'
    t_3['szport']='port3'
    t_4['szport']='port4'
    t_5['szport']='port5'
    t_6['szport']='port6'
    t_7['szport']='port7'
    t_8['szport']='port8'
    t_9['szport']='port9'

    t_output =pd.concat([t_S, t_B, t_SMB, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9])\
     .rename(columns={0:'t-stat', 1:'p-value'})
    return t_output

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

#各分组两种size大小,单位是千
#value weighted size
def vwsize(data):
    vwme=data.groupby(['jdate','szport']).apply(wavg, 'me','wt').to_frame().reset_index().rename(columns={0: 'vwme'})
    vwmedat = vwme.sort_values(by=['szport'])
    vwmedat2 = vwmedat.pivot(index='jdate', columns='szport', values='vwme')      
    vwmedat2 = vwmedat2.add_prefix('port')
    vwmedat2 = vwmedat2.rename(columns={'port1':'S', 'port10':'B'})
    vwme_mean = vwmedat2[['S', 'B', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
    vwme_mean = vwme_mean.rename(columns={0:'mean'}).reset_index()
    return vwme_mean
#equal weighted size
def ewsize(data):
    ewme=data.groupby(['jdate','szport'])['me'].mean().reset_index()
    ewmedat = ewme.sort_values(by=['szport'])
    ewmedat = ewmedat.pivot(index='jdate', columns='szport', values='me')      
    ewmedat = ewmedat.add_prefix('port')
    ewmedat = ewmedat.rename(columns={'port1':'S', 'port10':'B'})
    ewmedat_mean = ewmedat[['S', 'B', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
    ewmedat_mean = ewmedat_mean.rename(columns={0:'mean'}).reset_index()
    return ewmedat_mean

# value-weigthed return
def vwre(data):
    vwret=data.groupby(['jdate','szport']).apply(wavg, 'retadj','wt').to_frame().reset_index().rename(columns={0: 'vwret'})
    vwretdat = vwret.sort_values(by=['szport'])
    vwretdat2 = vwretdat.pivot(index='jdate', columns='szport', values='vwret')      
    vwretdat2 = vwretdat2.add_prefix('port')
    vwretdat2 = vwretdat2.rename(columns={'port1':'S', 'port10':'B'})
    vwretdat2['SMB'] = vwretdat2['S'] - vwretdat2['B']
    vwret_mean = vwretdat2[['S', 'B', 'SMB', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
    vwret_mean = vwret_mean.rename(columns={0:'mean'}).reset_index()
    sz_vw_output = pd.merge(vwret_mean, sz_t(vwretdat2), on=['szport'], how='inner') 
    return sz_vw_output

#equal weighted return
def ewre(data):
    ewret = data.groupby(['jdate','szport'])['retadj'].mean().reset_index()
    ewretdat = ewret.sort_values(by=['szport'])
    ewretdat2 = ewretdat.pivot(index='jdate', columns='szport', values='retadj')      
    ewretdat2 = ewretdat2.add_prefix('port')
    ewretdat2 = ewretdat2.rename(columns={'port1':'S', 'port10':'B'})
    ewretdat2['SMB'] = ewretdat2['S'] - ewretdat2['B']
    ewret_mean = ewretdat2[['S', 'B', 'SMB', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
    ewret_mean = ewret_mean.rename(columns={0:'mean'}).reset_index()
    sz_ew_output = pd.merge(ewret_mean, sz_t(ewretdat2), on=['szport'], how='inner')
    return sz_ew_output

#这里去按分配时间段调用，ccm5是全部的月度数据，用ccm5先根据ffyear分组然后调用上面的函数
# conutfirm()
# vwsize()
# ewsize()
# vwre()
# ewre()

############################
# table 2                  #
############################
ccm2_jun = ccm1_jun[['jdate','shrcd','exchcd','retadj','me','wt','cumretx','mebase','lme','dec_me','gvkey','datadate','yearend','be','count','oiadp','dvt','at','beme','posbm','szport','nonmissport']]
ccm3_jun = ccm2_jun[['jdate','oiadp','dvt','me','be','at','gvkey']]
ccm3_jun=ccm3_jun.sort_values(by=['gvkey','jdate'])

ccm3_jun['fea']=ccm3_jun.groupby(['gvkey'])['oiadp'].shift(-1)
ccm3_jun['fea_at']=ccm3_jun['fea']/ccm3_jun['at']
ccm3_jun['lat']=ccm3_jun.groupby(['gvkey'])['at'].shift(1)
ccm3_jun['earn_lat']=ccm3_jun['oiadp']/ccm3_jun['lat']
ccm3_jun['me_at']=ccm3_jun['me']/ccm3_jun['at']
ccm3_jun['dvt_be']=ccm3_jun['dvt']/ccm3_jun['be']
ccm3_jun['DD']=np.where(ccm3_jun['dvt']>0, 0, 1)
ccm3_jun['year']=ccm3_jun['jdate'].dt.year
ccm3_jun = ccm3_jun.dropna()
ccm3_jun=ccm3_jun.sort_values(by=['year','gvkey'])
#回归函数
def regress(group, formula):
    result = sm.formula.ols(formula, missing='drop', data = group).fit()
    return result.params

def regress2(df, dep, indep):
    Y = df[dep]
    X = df[indep]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model.params
#1994年回归报错
coef = ccm3_jun.groupby(['year']).apply(regress,'fea_at ~ me_at + DD + dvt_be + earn_lat')