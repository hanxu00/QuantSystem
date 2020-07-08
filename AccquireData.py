import tushare as ts
import pymongo as mg

class AccquireData:
    def __init__ (self):
        client = mg.MongoClient(host='localhost',port=27017)
        self.db = client.QuantTest

    def accquire_index(self,start_date=None,end_date=None):
        codes = ['000001.SH','000300.SH','399001.SZ','399005.SZ','399006.SZ']
        #codes = ['000001.SH']
        update_requests = []
        #未建立索引
        dailyIndex = 0
        for code in codes:
            df_daily = ts.pro_bar(ts_code=code,start_date=start_date,end_date=end_date,asset='I')
            for index in df_daily.index:
                doc = dict(df_daily.loc[index])
                #print(doc,flush=True)
                update_requests.append(
                    mg.UpdateOne(
                        {'ts_code':doc['ts_code'],'trade_date':doc['trade_date']},
                        {'$set':doc},
                        upsert=True
                    )
                )
            
            if len(update_requests) > 0:
                update_result = self.db.daily.bulk_write(update_requests,ordered=False)
                print('Save index daily, code: %s, inserted: %d,modified: %d' 
                    %(code, update_result .upserted_count, update_result.modified_count),
                    flush=True)

            #有数据结构以后建索引
            if dailyIndex == 0:
                self.db.daily.create_index([("ts_code", mg.ASCENDING),("trade_date", mg.ASCENDING)])
                dailyIndex = dailyIndex + 1
                for index in self.db.daily.list_indexes():
                    print(index)
            
    def accquire_stock(self, adj=None, start_date=None, end_date=None):
        #调用basic接口获取所有股票列表
        df_stock_L = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        df_stock_N = pro.stock_basic(exchange='', list_status='N', fields='ts_code,symbol,name,area,industry,list_date')
        df_stock_P = pro.stock_basic(exchange='', list_status='P', fields='ts_code,symbol,name,area,industry,list_date')
        df_stock = df_stock_L.append(df_stock_N).append(df_stock_P)
        #print(df_stock.head())
        codes = list(df_stock.ts_code)
        #codes = ['000001.SZ','600000.SH']
        update_requests = []
        #未建立索引
        #dailyIndex = 0
        for code in codes:
            df_daily = ts.pro_bar(ts_code=code, adj=None, start_date=start_date, end_date=end_date, asset='E')
            for index in df_daily.index:
                doc = dict(df_daily.loc[index])
                #print(doc,flush=True)
                update_requests.append(
                    mg.UpdateOne({'ts_code':doc['ts_code'],'trade_date':doc['trade_date']},
                        {'$set':doc},
                        upsert=True)
                    )
            
            if len(update_requests) > 0:
                collection_name = 'daily_hfq' if adj == 'hfq' else 'daily'
                update_result = self.db[collection_name].bulk_write(update_requests,ordered=False)
                print('Save index daily, code: %s, inserted: %d,modified: %d' 
                    %(code, update_result .upserted_count, update_result.modified_count),
                    flush=True)

            #有数据结构以后建索引
            # if dailyIndex == 0:
            #     self.daily.create_index([("ts_code", mg.ASCENDING),("trade_date", mg.ASCENDING)])
            #     dailyIndex = dailyIndex + 1
            #     for index in self.daily.list_indexes():
            #         print(index)


if __name__ == "__main__":   
    #设置token
    ts.set_token('38410c2bb386f08285e2b6f6421e7fd296f679594d2cd0fb330467cf')
    #初始化pro接口
    pro = ts.pro_api()
    ad = AccquireData()
    ad.accquire_index(start_date='20190101',end_date='20190131')
    ad.accquire_stock(start_date='20190101',end_date='20190131')
    ad.accquire_stock(adj='hfq',start_date='20190101',end_date='20190131')

'''
{
    'ts_code': '000001.SH',
    'trade_date': '20190102',
    'close': 2465.291,
    'open': 2497.8805,
    'high': 2500.2783,
    'low': 2456.4233,
    'pre_close': 2493.8962,
    'change': -28.6052,
    'pct_chg': -1.147,
    'vol': 109932013.0,
    'amount': 97592572.0
}
'''