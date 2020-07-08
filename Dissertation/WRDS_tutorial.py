import wrds

db = wrds.Connection(wrds_username='hanxu00')

#本地化用户名及密码，仅初次使用时运行
#db.create_pgpass_file()
#断开wrds链接
#db.close()

############ wrds tutorial #############
#https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/querying-wrds-data-python/
# db.close()
# db.connection()
# db.describe_table()
# db.get_table()
# db.list_tables()
# db.raw_sql()
# db.get_row_count()
# db.list_libraries()
# help(db.get_table)
# help(db.raw_sql)

#限制获取数据大小 get_table() raw_sql
# db.get_table('djones', 'djdaily', columns=['date', 'dji'], obs=10)
# db.raw_sql('select date,dji from djones.djdaily LIMIT 10;', date_cols=['date'])

db.list_libraries()
db.list_tables(library="wrdsapps")
db.describe_table(library="wrdsapps", table="firm_ratio")

data = db.get_table(library='djones', table='djdaily', columns=['date', 'dji'], obs=10)
data

data = db.raw_sql('select date,dji from djones.djdaily LIMIT 10;', date_cols=['date'])
data
data.head()

data = db.get_row_count('djones', 'djdaily')
data

db.raw_sql("select a.gvkey, a.datadate, a.tic, a.conm, a.at, a.lt, b.prccm, b.cshoq from comp.funda a join comp.secm b on a.gvkey = b.gvkey and a.datadate = b.datadate where a.tic = 'IBM' and a.datafmt = 'STD' and a.consol = 'C' and a.indfmt = 'INDL'")
#给SQL语句传参
parm = {'tickers': ('0015B', '0030B', '0032A', '0033A', '0038A')}
data = db.raw_sql('SELECT datadate,gvkey,cusip FROM comp.funda WHERE tic in %(tickers)s', params=parm)

db.close()