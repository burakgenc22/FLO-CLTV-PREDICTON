#### KÜTÜPHANELERİN IMPORT EDİLMESİ ####

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from sklearn.preprocessing import MinMaxScaler

#### VERİ SETİNİN OKUTULMASI VE HAZIRLANMASI ####

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()
df.describe().T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

df.info
df.dtypes
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.dtypes

#### CLTV VERİ YAPISININ OLUŞTURULMASI ####

analysis_date = dt.datetime(2021, 6, 1)
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_week"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) /7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_cltv_avg"] = df["total_price"] / df["total_order"]
cltv_df.head()

#### BG-NBD ve GAMMA-GAMMA MODELLERİNİN KURULMASI ve CLTV'NİN HESAPLANMASI ####

## bg-nbd modelinin kurulması ve hesaplamalar ##
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_week"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın alma tahminleri #
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                  cltv_df["frequency"],
                                                                  cltv_df["recency_cltv_week"],
                                                                  cltv_df["T_weekly"]
                                                                  )
# 6 ay içerisinde müşterilerden beklenen satın alma tahminleri #
cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                                  cltv_df["frequency"],
                                                                  cltv_df["recency_cltv_week"],
                                                                  cltv_df["T_weekly"]
                                                                  )
cltv_df.head()

## gamma-gamma modelinin kurulması ##

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])
cltv_df.head()


#### CLTV DEĞERLERİNE GÖRE SEGMENTLERE AYIRILMASI ####

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency_cltv_week"],
                                   cltv_df["T_weekly"],
                                   cltv_df["monetary_cltv_avg"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.head()

cltv_df.to_excel("flo_cltv_prediction.xlsx")