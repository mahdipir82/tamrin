import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#===================================
#واکشی فایل 
dataset=pd.read_csv('tamrin/loans.csv')
#==================================
#نمایش تعداد مقدار های خالی
# print(dataset.isnull().sum())
#==================================
#نمایش داده های پرت و جایگزین ان با  
def chart_show(data):
    plt.boxplot(data)
    plt.show()

# chart_show(dataset['loan_amount'])
# chart_show(dataset['rate'])
dataset=dataset[(dataset['rate']>=0) & (dataset['rate']<= 9)]
#=======================================================
### رمزگزاری داده ها 
# # label_encoder
def label_encoder(data, columns):
    le = LabelEncoder()
    for loan_type in columns:
        data[loan_type] = le.fit_transform(data[loan_type])
    return data
dataset=label_encoder(dataset,['loan_type'])
#=======================================================
#حذف ستون های غیر مورد نیاز
def drop_columns(data, columns):
    for col in columns:
        data.drop(col, axis=1, inplace=True)
    return data
dataset=drop_columns(dataset,['repaid','loan_start','loan_end'])
#===================================
# نرمال سازی
from sklearn.preprocessing import MinMaxScaler
def min_max_scaler(data, columns):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = columns
    return data
#==============================================================
# استاندار سازی
from sklearn.preprocessing import StandardScaler
def standard_scaler(data, columns):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)
    data.columns = columns
    return data
#================================================================
dataset1=min_max_scaler(dataset,['client_id','loan_type','loan_amount','loan_id','rate'])
dataset=standard_scaler(dataset,['client_id','loan_type','loan_amount','loan_id','rate'])
print(dataset1)
print(dataset)