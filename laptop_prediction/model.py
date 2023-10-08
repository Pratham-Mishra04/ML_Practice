import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

df = pd.read_csv("dataset.csv")

df = df[df['Price']<180000]
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df['Ram'] = (df['Ram'].str.replace('GB', '')).astype('int32')
df['Weight'] = (df['Weight'].str.replace('kg', '')).astype('float')

df['touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df['ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

resolution_split = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['x_res'] = resolution_split[0]
df['y_res'] = resolution_split[1]
df['x_res'] = df['x_res'].str.replace(',','').str.findall(r'\d+\.?\d+').apply(lambda x:x[0])
df['x_res']=df['x_res'].astype('int32')
df['y_res']=df['y_res'].astype('int32')

df['uhd'] = df['x_res'].apply(lambda x:1 if x>=3840 else 0)
df['qhd'] = df['x_res'].apply(lambda x:1 if x<3840 and x>1920 else 0)
df['fhd'] = df['x_res'].apply(lambda x:1 if x<=1920 else 0)

df['ppi'] = (((df['x_res']**2) + (df['y_res']**2))**0.5)/df['Inches'].astype('int32')
df.drop(columns=['ScreenResolution', 'x_res', 'y_res', 'Inches'], inplace=True)

df['CPU Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

def get_processor(x):
    if x=="Intel Core i5" or x=='Intel Core i7' or x=='Intel Core i3':
        return x
    elif x.split()[0] == 'Intel':
        return "Other Intel Processor"
    else:
        return "AMD Processor"

def get_clock_speed(x):
    y = re.findall(r'\d\.?\d+GHz', x)
    if len(y) > 0:
        return y[0].replace('GHz', '')
    else:
        return 0.0
    
df['cpu'] = df['CPU Name'].apply(get_processor)
df['clock_speed'] = df['Cpu'].apply(get_clock_speed).astype("float32")
df.drop(columns=['Cpu', 'CPU Name'], inplace=True)

df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.extract('(\d+)')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '', regex=True)

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["hdd"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["ssd"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["flash_storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage', 'Memory'],inplace=True)


df['gpu'] = df['Gpu'].apply(lambda x: x.split()[0])
df=df[df['gpu']!="ARM"]
df.drop('Gpu', inplace=True, axis=1)

def get_os(x):
    if "Windows" in x:
        return "Windows"
    elif x=="macOS" or x=="Mac OS X":
        return "Mac"
    elif x=="Linux" or x=="No OS":
        return x
    else :
        return "Other"

df['os'] = df['OpSys'].apply(get_os)
df.drop('OpSys', axis=1, inplace=True)

df.rename(columns={"Company":"company", "TypeName":"type", "Ram":"ram", "Weight":"weight", "Price":"price"}, inplace=True)

df.drop('hdd', axis=1, inplace=True)
df.drop('hybrid', axis=1, inplace=True)
df.drop('flash_storage', axis=1, inplace=True)
df.drop('fhd', axis=1, inplace=True)

le = LabelEncoder()

X = df.drop('price', axis=1)
y = np.log(df['price'])

X['company'] = le.fit_transform(X['company'])
X['type'] = le.fit_transform(X['type'])
X['os'] = le.fit_transform(X['os'])
X['cpu'] = le.fit_transform(X['cpu'])
X['gpu'] = le.fit_transform(X['gpu'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = ExtraTreesRegressor(n_estimators=100, max_depth=30, min_samples_split=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMAE: ", mse)
print("MSE: ", mse)
print("RMSE: ", np.sqrt(mse))
print("\nR^2: ", r2)

dump(model, 'model.joblib') 