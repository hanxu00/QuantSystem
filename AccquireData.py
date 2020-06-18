import tushare as ts
#设置token
ts.set_token('38410c2bb386f08285e2b6f6421e7fd296f679594d2cd0fb330467cf')
#初始化pro接口
pro = ts.pro_api()
df_daily = ts.pro_bar(ts_code='000001',start_date='20150101', end_date='20150105',asset='E')
print(df_daily)