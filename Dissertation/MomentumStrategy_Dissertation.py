#2020.7.11 删除多余代码及注释，详细中文注释见sample
#2020.7.13 加入skip month
#2020.8.11 加入市场状态；分四个时间段，避免内存占用过大无法运行的问题

import pandas as pd
import numpy as np
import wrds
from pandas.tseries.offsets import *
from scipy import stats

class Momentum:
    def accquireData(self,beginDate,endDate):
        ## 1.数据获取及清洗
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
                            where a.date between \'""" + beginDate + '''\' and \'''' + endDate +"""\' 
                            and b.exchcd between -2 and 2
                            and b.shrcd between 10 and 11
                            """) 
        conn.close()
        # Change variable format to int
        crsp[['permco','permno','shrcd','exchcd']]=\
            crsp[['permco','permno','shrcd','exchcd']].astype(int)
        # Line up date to be end of month
        crsp['date']=pd.to_datetime(crsp['date'])
        print('crsp数据集的行列数：\n',crsp.shape)
        return crsp
        
    def accquireMarket(self):       
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

        #ff3d = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/F-F_Research_Data_Factors_daily.CSV', 
                                    #parse_dates={'date':[0]}, encoding='utf8', header=0)
        ff3d = pd.read_csv('F-F_Research_Data_Factors_daily.CSV',parse_dates={'date':[0]}, encoding='utf8', header=0)                                           
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
        market = market.drop(['date'],axis=1)
        #marketdat = market.groupby(['state'])['mret'].describe()[['count','mean', 'std']]
        #voladat = market.groupby(['state'])['vola'].describe()[['count','mean']]
        print('market数据集的行列数：\n',market.shape)
        return market

    def calReturn(self,crsp_m,J,S,K):
        ## 2.Create Momentum Portfolio 
        #J Formation Period, S skipping period
        _tmp_crsp = crsp_m[['permno','date','ret']].sort_values(['permno','date'])\
            .set_index('date')
        #print('数据集中是否存在缺失值：\n',any(_tmp_crsp['ret'].isnull()))
        _tmp_crsp['ret']=_tmp_crsp['ret'].fillna(0)
        #print('数据集中是否存在缺失值：\n',any(_tmp_crsp['ret'].isnull()))
        _tmp_crsp['logret']=np.log(1+_tmp_crsp['ret'])
        umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(J, min_periods=J).sum()
        umd = umd.reset_index()
        umd['cumret']=np.exp(umd['logret'])-1
        
        ## 3.Formation of 10 Momentum Portfolios
        # K Holding Period Length
        # For each date: assign ranking 1-10 based on cumret
        # 1=lowest 10=highest cumret
        umd=umd.dropna(axis=0, subset=['cumret'])
        umd['rank'] = umd.groupby('date')['cumret'].rank(method='first')        
        umd['momr']=umd.groupby('date')['rank'].transform(lambda x: pd.qcut(x, 10, labels=False))
        umd.momr=umd.momr.astype(int)
        umd['momr'] = umd['momr']+1
        # form_date是formation的最后一个交易日
        # medate是formation自然月的最后一天
        # hdate1是holding period开始月的第一天
        # hdate2是holding period结束月的最后一天
        # 构建hdate1和hdate2是为了后面merge以后保留holding period的数据
        ## 本身把S放在medate里，但是monthEnd可能有bug，对10066股票，总有两个月生成一样的medate，所以S改到hdate1里了
        umd['form_date'] = umd['date']
        umd['medate'] = umd['date']+MonthEnd(0)      
        umd['hdate1']=umd['medate']+MonthBegin(S+1)   
        umd['hdate2']=umd['medate']+MonthEnd(S+K)
        umd = umd[['permno', 'form_date','momr','hdate1','hdate2']]
        print('umd的行列数：\n',umd.shape)
        # join rank and return data together
        # note: this step consumes a lot of memory so takes a while
        _tmp_ret = crsp_m[['permno','date','ret']]
        port = pd.merge(_tmp_ret, umd, on=['permno'], how='inner')
        # merge是按permno来的，所以_tmp_ret里每个permno和date后面都跟了umd里每个permno里的数据，按下面这个规则可以剔除
        # 规则是hdate1<=date<=hdate2,也就是说保留在holding period月份中的股票数据
        port = port[(port['hdate1']<=port['date']) & (port['date']<=port['hdate2'])]
        print('port的行列数：\n',port.shape)

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
        ewretdat3 = ewretdat3.reset_index()
        # ewretdat3['cumret_2']=ewretdat3['1+2'].cumprod()-1
        # ewretdat3['cumret_3']=ewretdat3['1+3'].cumprod()-1
        # ewretdat3['cumret_4']=ewretdat3['1+4'].cumprod()-1
        # ewretdat3['cumret_5']=ewretdat3['1+5'].cumprod()-1
        # ewretdat3['cumret_6']=ewretdat3['1+6'].cumprod()-1
        # ewretdat3['cumret_7']=ewretdat3['1+7'].cumprod()-1
        # ewretdat3['cumret_8']=ewretdat3['1+8'].cumprod()-1
        # ewretdat3['cumret_9']=ewretdat3['1+9'].cumprod()-1
        #ewretdat3.head()
        #ewretdat3.to_csv('cumret_( %d,1,1)_1991-2019.csv' %J)
        print('ewretdat3的行列数：\n',ewretdat3.shape)
        return ewretdat3

    def WMLoutput(self,ewretdat3):
        ## 5.Portfolio Summary Statistics
        # Mean 
        #mom_mean = ewretdat3[['winners', 'losers', 'long_short', 'port2', 'port3', 'port4', 'port5', 'port6', 'port7', 'port8', 'port9']].mean().to_frame()
        mom_mean = ewretdat3[['winners', 'losers', 'long_short']].mean().to_frame()
        mom_mean = mom_mean.reset_index().rename(columns={0:'mean','index':'momr'})
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
        print(mom_output)
        return mom_output


if __name__ == "__main__":
    mm = Momentum()
    # firmData1 = mm.accquireData(beginDate='01/01/1926',endDate='12/31/1955')
    # firmData2 = mm.accquireData(beginDate='01/01/1950',endDate='12/31/1985')
    # firmData3 = mm.accquireData(beginDate='01/01/1980',endDate='12/31/2000')
    # firmData4 = mm.accquireData(beginDate='01/01/1995',endDate='06/30/2020')
    
    firmData = mm.accquireData(beginDate='01/01/1988',endDate='01/31/2019')
    
    marketData = mm.accquireMarket()
    #设定不同的period
    #formationPeriod = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    formationPeriod = [1,2,3,4,5,6,7,8,9,10,11]
    #formationPeriod = [1]
    holdingPeriod = [1]
    SkipMonth = 1
    #SkipMonth = 0
    #设定不同的marketstate
    #marketState = ['Bull','Correction','Bear','Rebound']
    #marketState = ['Rebound']
    #for ms in marketState:
    output = pd.DataFrame()
    for m in formationPeriod:
        for n in holdingPeriod:
            # portReturn1 = mm.calReturn(crsp_m=firmData1,J=m,S=SkipMonth,K=n) #按J计算累计收益,构建收益分位数组合，输出各组合收益统计量
            # portReturn2 = mm.calReturn(crsp_m=firmData2,J=m,S=SkipMonth,K=n)
            # portReturn3 = mm.calReturn(crsp_m=firmData3,J=m,S=SkipMonth,K=n)
            # portReturn4 = mm.calReturn(crsp_m=firmData4,J=m,S=SkipMonth,K=n)
            # portReturnTemp1 = pd.concat([portReturn1,portReturn2])
            # portReturnTemp2 = pd.concat([portReturn3,portReturn4])
            # portReturn = pd.concat([portReturnTemp1,portReturnTemp2])
            
            portReturn = mm.calReturn(crsp_m=firmData,J=m,S=SkipMonth,K=n)
            
            portReturn['monthEndDate'] = portReturn['date'] + MonthEnd(0)
            portReturn = portReturn.drop_duplicates(subset=['monthEndDate'])
            portReturn_m = pd.merge(portReturn, marketData, on=['monthEndDate'], how='left')       
            
            #-------这部分是分市场状态--------
            # if ms == 'Bull':
            #     portReturn_m.to_csv('portReturn_%d.csv' %(m))
            # portReturn_ms = portReturn_m.loc[portReturn_m.state == ms,]
            # print('portReturn_ms的行列数：\n',portReturn_ms.shape)
            # #portReturn_ms.to_csv('portReturn_%s.csv' %(ms))
            # output_tmp = mm.WMLoutput(ewretdat3=portReturn_ms) #输出组合收益统计量
            #-------------------------

            output_tmp = mm.WMLoutput(ewretdat3=portReturn_m)
            
            if output.empty:
                output = output_tmp
            else:
                output = pd.concat([output,output_tmp])
            #print("formationPeriod: %d, holdingperiod: %d, state: %s" %(m,n,ms))
    #output.to_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/output_%s.csv' %(ms))
    #output.to_csv('output_%s.csv' %(ms))
    #output.to_csv('output_(1-11,1,1)_1929-2019.csv')
    output.to_csv('output_(1-11,1,1)_1990-2019.csv')

