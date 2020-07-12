#2020.7.11 删除多余代码及注释，详细中文注释见sample

import pandas as pd
import numpy as np
import wrds
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from scipy import stats

class Momentum:
    def accquireData(self):
        ## 1.数据获取及清洗
        # Connect to WRDS
        conn=wrds.Connection(wrds_username='hanxu00')
        # AcquireData 
        crsp_m = conn.raw_sql("""
                            select a.permno, a.permco, b.ncusip, a.date, 
                            b.shrcd, b.exchcd, b.siccd,
                            a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
                            from crsp.msf as a
                            left join crsp.msenames as b
                            on a.permno=b.permno
                            and b.namedt<=a.date
                            and a.date<=b.nameendt
                            where a.date between '01/01/1965' and '12/31/1990'
                            and b.exchcd between -2 and 2
                            and b.shrcd between 10 and 11
                            """) 
        conn.close()
        print('数据集的行列数：\n',crsp_m.shape)
        # Change variable format to int
        crsp_m[['permco','permno','shrcd','exchcd']]=\
            crsp_m[['permco','permno','shrcd','exchcd']].astype(int)
        # Line up date to be end of month
        crsp_m['date']=pd.to_datetime(crsp_m['date'])
        return crsp_m

    def calReturn(self,crsp_m,J,K):
        ## 2.Create Momentum Portfolio 
        #J = 6 # Formation Period Length: J can be between 3 to 12 months
        #K = 6 # Holding Period Length: K can be between 3 to 12 months
        _tmp_crsp = crsp_m[['permno','date','ret']].sort_values(['permno','date'])\
            .set_index('date')
        #print('数据集中是否存在缺失值：\n',any(_tmp_crsp['ret'].isnull()))
        _tmp_crsp['ret']=_tmp_crsp['ret'].fillna(0)
        #print('数据集中是否存在缺失值：\n',any(_tmp_crsp['ret'].isnull()))
        # Calculate rolling cumulative return
        _tmp_crsp['logret']=np.log(1+_tmp_crsp['ret'])
        umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(J, min_periods=J).sum()
        #reset_index行索引设置为变量
        umd = umd.reset_index()
        umd['cumret']=np.exp(umd['logret'])-1
        umd.head()
        return umd
        
    def createMP(self,umd,crsp_m,K):
        ## 3.Formation of 10 Momentum Portfolios
        # For each date: assign ranking 1-10 based on cumret
        # 1=lowest 10=highest cumret
        umd=umd.dropna(axis=0, subset=['cumret'])
        umd['momr']=umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))
        umd.momr=umd.momr.astype(int)
        umd['momr'] = umd['momr']+1
        umd.head()
        # Corrected previous version month end line up issue
        # First lineup date to month end date medate
        # Then calculate hdate1 and hdate2 using medate
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
        # Skip first two years of the sample 
        start_yr = umd3['date'].dt.year.min()+2
        umd3 = umd3[umd3['date'].dt.year>=start_yr]
        umd3 = umd3.sort_values(by=['date','momr'])
        # Create one return series per MOM group every month
        ewret = umd3.groupby(['date','momr'])['ret'].mean().reset_index()
        ewstd = umd3.groupby(['date','momr'])['ret'].std().reset_index()
        ewret = ewret.rename(columns={'ret':'ewret'})
        ewstd = ewstd.rename(columns={'ret':'ewretstd'})
        ewretdat = pd.merge(ewret, ewstd, on=['date','momr'], how='inner')
        ewretdat = ewretdat.sort_values(by=['momr'])
        # portfolio summary
        ewretdat.groupby(['momr'])['ewret'].describe()[['count','mean', 'std']]
        return ewretdat

    def WMLoutput(self,ewretdat):
        ## 4.Long-Short Portfolio Returns
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
        ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
        ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
        ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1
        ewretdat3.head()

        ## 5.Portfolio Summary Statistics
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
        #print(mom_output)
        return mom_output


if __name__ == "__main__":
    mm = Momentum()
    rawData = mm.accquireData() 
    formationPeriod = [3,6,9,12]
    holdingPeriod = [3,6,9,12]
    #formationPeriod = [3,6]
    #holdingPeriod = [3]
    output = pd.DataFrame()
    for m in formationPeriod:
        for n in holdingPeriod:
            cumReturn = mm.calReturn(crsp_m=rawData,J=m,K=n) #按JK规则计算累计收益
            portfolioSummary = mm.createMP(umd=cumReturn,crsp_m=rawData,K=n) #构建收益分位数组合，输出各组合收益统计量
            output_tmp = mm.WMLoutput(ewretdat=portfolioSummary) #构建WML组合，输出组合收益统计量
            if output.empty:
                output = output_tmp
            else:
                output = pd.concat([output,output_tmp])
            print("formationPeriod: %d, holdingperiod: %d" %(m,n))
    print(output)
    output.to_excel(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/output.xlsx')

