#2020.7.11 删除多余代码及注释，详细中文注释见sample
#2020.7.13 加入skip month

import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
from scipy import stats

class Momentum:
    def accquireData(self):
        ## 1.数据获取及清洗
        # Connect to WRDS
        conn=wrds.Connection(wrds_username='hanxu00')
        # AcquireData 1965-1990
        # crsp_m = conn.raw_sql("""
        #                     select a.permno, a.permco, b.ncusip, a.date, 
        #                     b.shrcd, b.exchcd, b.siccd,
        #                     a.ret, a.vol, a.shrout, a.prc, a.cfacpr, a.cfacshr
        #                     from crsp.msf as a
        #                     left join crsp.msenames as b
        #                     on a.permno=b.permno
        #                     and b.namedt<=a.date
        #                     and a.date<=b.nameendt
        #                     where a.date between '01/01/1965' and '12/31/1990'
        #                     and b.exchcd between -2 and 2
        #                     and b.shrcd between 10 and 11
        #                     """) 
        # data 1991-now
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
        print('数据集的行列数：\n',crsp_m.shape)
        # Change variable format to int
        crsp_m[['permco','permno','shrcd','exchcd']]=\
            crsp_m[['permco','permno','shrcd','exchcd']].astype(int)
        # Line up date to be end of month
        crsp_m['date']=pd.to_datetime(crsp_m['date'])
        return crsp_m

    def calReturn(self,crsp_m,J):
        ## 2.Create Momentum Portfolio 
        #J Formation Period, S skipping period
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

    def createMP(self,umd,crsp_m,S,K):
        ## 3.Formation of 10 Momentum Portfolios
        # K Holding Period Length
        # For each date: assign ranking 1-10 based on cumret
        # 1=lowest 10=highest cumret
        umd=umd.dropna(axis=0, subset=['cumret'])
        umd['momr']=umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))
        umd.momr=umd.momr.astype(int)
        umd['momr'] = umd['momr']+1
        # form_date是formation的最后一个交易日
        # medate是formation自然月的最后一天
        # hdate1是holding period开始月的第一天
        # hdate2是holding period自然月的最后一天
        # 构建hdate1和hdate2是为了后面merge以后保留holding period的数据
        ## 本身把S放在medate里，但是monthEnd可能有bug，对10066股票，总有两个月生成一样的medate，所以S改到hdate1里了
        umd['form_date'] = umd['date']
        umd['medate'] = umd['date']+MonthEnd(0)
        umd['hdate1']=umd['medate']+MonthBegin(S+1)   
        umd['hdate2']=umd['medate']+MonthEnd(S+K)
        umd = umd[['permno', 'form_date','momr','hdate1','hdate2']]
        # join rank and return data together
        # note: this step consumes a lot of memory so takes a while
        _tmp_ret = crsp_m[['permno','date','ret']]
        port = pd.merge(_tmp_ret, umd, on=['permno'], how='inner')
        # merge是按permno来的，所以_tmp_ret里每个permno和date后面都跟了umd里每个permno里的数据，按下面这个规则可以剔除
        # 规则是hdate1<=date<=hdate2,也就是说保留在holding period月份中的股票数据
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

        # ewretdat3['1+2'] = 1+ewretdat3['port2']
        # ewretdat3['1+3'] = 1+ewretdat3['port3']
        # ewretdat3['1+4'] = 1+ewretdat3['port4']
        # ewretdat3['1+5'] = 1+ewretdat3['port5']
        # ewretdat3['1+6'] = 1+ewretdat3['port6']
        # ewretdat3['1+7'] = 1+ewretdat3['port7']
        # ewretdat3['1+8'] = 1+ewretdat3['port8']
        # ewretdat3['1+9'] = 1+ewretdat3['port9']

        ewretdat3['cumret_winners']=ewretdat3['1+winners'].cumprod()-1
        ewretdat3['cumret_losers']=ewretdat3['1+losers'].cumprod()-1
        ewretdat3['cumret_long_short']=ewretdat3['1+ls'].cumprod()-1

        # ewretdat3['cumret_2']=ewretdat3['1+2'].cumprod()-1
        # ewretdat3['cumret_3']=ewretdat3['1+3'].cumprod()-1
        # ewretdat3['cumret_4']=ewretdat3['1+4'].cumprod()-1
        # ewretdat3['cumret_5']=ewretdat3['1+5'].cumprod()-1
        # ewretdat3['cumret_6']=ewretdat3['1+6'].cumprod()-1
        # ewretdat3['cumret_7']=ewretdat3['1+7'].cumprod()-1
        # ewretdat3['cumret_8']=ewretdat3['1+8'].cumprod()-1
        # ewretdat3['cumret_9']=ewretdat3['1+9'].cumprod()-1
        ewretdat3.head()

        ## 5.Portfolio Summary Statistics
        # Mean 
        #mom_mean = ewretdat3[['winners', 'losers', 'long_short', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
        mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
        mom_mean = mom_mean.rename(columns={0:'mean'}).reset_index()
        mom_mean
        # T-Value and P-Value
        # DataFrame.T: Transpose index and columns.
        t_losers = pd.Series(stats.ttest_1samp(ewretdat3['losers'],0.0)).to_frame().T
        t_winners = pd.Series(stats.ttest_1samp(ewretdat3['winners'],0.0)).to_frame().T
        t_long_short = pd.Series(stats.ttest_1samp(ewretdat3['long_short'],0.0)).to_frame().T

        # t_2 = pd.Series(stats.ttest_1samp(ewretdat3['port2'],0.0)).to_frame().T
        # t_3 = pd.Series(stats.ttest_1samp(ewretdat3['port3'],0.0)).to_frame().T
        # t_4 = pd.Series(stats.ttest_1samp(ewretdat3['port4'],0.0)).to_frame().T
        # t_5 = pd.Series(stats.ttest_1samp(ewretdat3['port5'],0.0)).to_frame().T
        # t_6 = pd.Series(stats.ttest_1samp(ewretdat3['port6'],0.0)).to_frame().T
        # t_7 = pd.Series(stats.ttest_1samp(ewretdat3['port7'],0.0)).to_frame().T
        # t_8 = pd.Series(stats.ttest_1samp(ewretdat3['port8'],0.0)).to_frame().T
        # t_9 = pd.Series(stats.ttest_1samp(ewretdat3['port9'],0.0)).to_frame().T

        t_losers['momr']='losers'
        t_winners['momr']='winners'
        t_long_short['momr']='long_short'

        # t_2['momr']='port2'
        # t_3['momr']='port3'
        # t_4['momr']='port4'
        # t_5['momr']='port5'
        # t_6['momr']='port6'
        # t_7['momr']='port7'
        # t_8['momr']='port8'
        # t_9['momr']='port9'

        t_output =pd.concat([t_winners, t_losers, t_long_short])\
            .rename(columns={0:'t-stat', 1:'p-value'})
        # t_output =pd.concat([t_winners, t_losers, t_long_short, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9])\
        #     .rename(columns={0:'t-stat', 1:'p-value'})
        t_output
        # Combine mean, t and p
        mom_output = pd.merge(mom_mean, t_output, on=['momr'], how='inner')
        #print(mom_output)
        return mom_output


if __name__ == "__main__":
    mm = Momentum()
    rawData = mm.accquireData()
    # formationPeriod = [3,6,9,12]
    # holdingPeriod = [3,6,9,12]
    # SkipMonth = 0
    formationPeriod = [11]
    holdingPeriod = [1]
    SkipMonth = 1
    output = pd.DataFrame()
    for m in formationPeriod:
        for n in holdingPeriod:
            cumReturn = mm.calReturn(crsp_m=rawData,J=m) #按J计算累计收益
            portfolioSummary = mm.createMP(umd=cumReturn,crsp_m=rawData,S=SkipMonth,K=n) #构建收益分位数组合，输出各组合收益统计量
            output_tmp = mm.WMLoutput(ewretdat=portfolioSummary) #构建WML组合，输出组合收益统计量
            if output.empty:
                output = output_tmp
            else:
                output = pd.concat([output,output_tmp])
            print("formationPeriod: %d, holdingperiod: %d" %(m,n))
    print(output)

    #output.to_excel(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/output_(J,S,K)1965-1990.xlsx')
    #output.to_excel(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/output_(J,S,K)1991-n.xlsx')
    output.to_excel(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/output_(11,1,1)1991-n.xlsx')

