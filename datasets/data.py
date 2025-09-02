import akshare as ak
import pandas as pd

'''
获取证券数据
'''
gupiao_list = [159790, 159568, 159845, 601360, 688360, 159529, 159601]
data_list = []
for gupiao in gupiao_list:
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=str(gupiao), period="daily", start_date="20001231",
                                            end_date='20250715',
                                            adjust="")
    stock_zh_a_hist_df.to_csv('raw_data/{}.csv'.format(str(gupiao)), index=False, encoding='gbk')
