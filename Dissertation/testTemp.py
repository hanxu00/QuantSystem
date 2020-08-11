crsp_m.head(24)
crsp_m.shape

ff3.head()
ff3.dtypes

rmrf.head()

rmrfmv.head()

rmrfMaxVola.head()
rmrfMaxVola.shape

rmrfMaxVola.head()

crsp_mv.head()
crsp_mv.dtypes
crsp_mv.shape

umd.head(20)
umd.shape

umd2.head()
umd2.shape

# umd.ix[2222:2250,:]
# umd.head()
# umd.shape
# umd2.shape
# umd2.head()
# umd3.shape
# ewret.head(20)
#ewstd.head(60)

ewretdat.head()
ewretdat2.head()
ewretdat3.head()
ewretdat3.shape


#-------------------------------------
#comp.to_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/SizeEffect/comp.csv', encoding='utf8')
#comp1 = pd.read_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/SizeEffect/comp.csv', index_col=0, encoding='utf8')
comp.head()
comp.shape
comp.dtypes

crsp_m.head()
crsp_m.shape

dlret.head()
dlret.shape

crsp.head()
crsp.shape

crsp2.head()
crsp2.shape

crsp3.head(30)
crsp3.shape
crsp3.dtypes

crsp_jun.head()
crsp_jun.shape

ccm.head()
ccm.shape
ccm_jun.head()
ccm_jun.shape

nyse.head()

ccm1_jun.tail(50)
ccm1_jun.dtypes

june.head(100)

ccm4.head(15)

ff_factors.head(20)

ccm4['exchcd'].describe()

vwme_mean.dtypes

ccm3_jun['year'].describe()
ccm3_jun.dtypes
ccm3_jun = ccm3_jun.dropna(axis = 0, subset=['fea_at','me_at','DD','dvt_be','earn_lat'])
ccm3_jun=ccm3_jun.dropna(axis = 1)
ccm3_jun.isnull
ccm3_jun=ccm3_jun[ccm3_jun.notnull()]
ccm3_jun.fillna(value = 0)
any(ccm5_jun.isnull())

ccm4_jun=ccm3_jun[ccm3_jun['year']==1993]
temp = regress(ccm4_jun,'fea_at ~ me_at + DD + dvt_be + earn_lat')
ccm4_jun

ccm5_jun=ccm3_jun[ccm3_jun['year']==1991]
ccm5_jun.dropna()
xxx=ccm4_jun['me_at']
xxx=ccm4_jun[['me_at']]

temp = regress(ccm4_jun,'fea_at',['me_at','DD','dvt_be','earn_lat'])
model = sm.formula.ols('fea_at ~ me_at + DD + dvt_be + earn_lat',missing='drop', data = ccm4_jun).fit()
model.params

crsp_m.head()
ff3.head()
ff3['date',1]

temp = crsp_m.loc[crsp_m.state == 'Bull',]

ewretdat3['monthEndDate'] = ewretdat3['date'] + MonthEnd(0)
ewretdat3 = pd.merge(ewretdat3, market, on=['monthEndDate'], how='left')
ewretdat3.dtypes
a='xx'
market.to_csv(r'/Users/hanxu/OneDrive - University of Bristol/10.Dissertation/Data/test_ %s.csv' %(a))



_tmp_crsp = crsp[['permno','date','ret']].sort_values(['permno','date'])\
    .set_index('date')
_tmp_crsp['ret']=_tmp_crsp['ret'].fillna(0)
_tmp_crsp['logret']=np.log(1+_tmp_crsp['ret'])
umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(5, min_periods=5).sum()
umd = umd.reset_index()
umd['cumret']=np.exp(umd['logret'])-1


umd=umd.dropna(axis=0, subset=['cumret'])

umd['rank'] = umd.groupby('date')['cumret'].rank(method='first')
#df['decile'] = pd.qcut(df['rank'].values, 10).codes

umd['momr']=umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, 10, labels=False))
umd['momr2']=umd.groupby('date')['rank'].transform(lambda x: pd.qcut(x, 10, labels=False))
umd.head(40)

umd.momr=umd.momr.astype(int)
umd['momr'] = umd['momr']+1

beginDate= '01/01/1990'
endDate= '12/31/1991'