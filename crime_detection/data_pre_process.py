import pandas as pd

path = './data/Crime_Data_from_2020_to_Present.csv'
df = pd.read_csv(path)
top_10_crime_codes = df['Crm Cd'].value_counts().nlargest(10).index
df_top_10 = df[df['Crm Cd'].isin(top_10_crime_codes)]
df_top_10['Date Rptd'] = pd.to_datetime(df_top_10['Date Rptd'])
df_top_10['DATE OCC'] = pd.to_datetime(df_top_10['DATE OCC'])
df_top_10['Month_Rptd'] = df_top_10['Date Rptd'].dt.month
df_top_10['Day_Rptd'] = df_top_10['Date Rptd'].dt.day
df_top_10['Year_Rptd'] = df_top_10['Date Rptd'].dt.year
df_top_10['Month_OCC'] = df_top_10['DATE OCC'].dt.month
df_top_10['Day_OCC'] = df_top_10['DATE OCC'].dt.day
df_top_10['Year_OCC'] = df_top_10['DATE OCC'].dt.year
df_top_10['Date Difference'] = (df_top_10['Date Rptd'] - df_top_10['DATE OCC']).dt.days
df_top_10['Hour'] = df_top_10['TIME OCC'] // 100
df_top_10['Minute'] = df_top_10['TIME OCC'] % 100
df_top_10['Vict Age'] = df_top_10['Vict Age'].apply(lambda x: None if x == 0 else x)
df_top_10['Crm Cd'] = df_top_10['Crm Cd'].astype('category')
df_top_10['Label'] = df_top_10['Crm Cd'].cat.codes
print(df_top_10)
df_top_10.to_csv('./data/Pre-processing_Crime_Data_Top10.csv', index=False)
