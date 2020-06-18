import tushare as ts
import pymongo as mg

class AccquireData:
    def __init__ (self):
        client = mg.MongoClient(host='localhost',port=27017)
        db = client.QuantTest
        self.daily = db.daily
        self.daily_hfq = db.daily_hfq

    def accquire_index(self,start_date=None,end_date=None):
        
        #codes = ['000001.SH','000300.SH','399001.SZ','399005.SZ','399006.SZ']
        codes = ['000001.SH']
        for code in codes:
            df_daily = ts.pro_bar(ts_code=code,start_date=start_date,end_date=end_date,asset='I')
            for index in df_daily.index:
                doc = dict(df_daily.loc[index])
                print(doc,flush=True)
        
        

    def accquire_stock(self,start_date=None,end_date=None):
        pass

if __name__ == '_main_':
    #设置token
    ts.set_token('38410c2bb386f08285e2b6f6421e7fd296f679594d2cd0fb330467cf')
    #初始化pro接口
    pro = ts.pro_api()
    ad = AccquireData()
    ad.accquire_index(start_date='20190101',end_date='20190131')
    ad.accquire_stock(start_date='20190101',end_date='20190131')