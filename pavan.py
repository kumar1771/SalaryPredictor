import pandas as pd
df=pd.read_csv(r"C:\Users\UGANDA\Downloads\iris.csv")
print(df[0:5])
print(df.info())        # Summary info about the dataset
print(df.describe())
c=1
l=0
k=df['species'][1]
for i in range(1,len(df)):
    if k==df['species'][i]:
        c=c+1
        l+=df['sepal_length'][i]
    else:
        print(k,c)
        print(l//c)
        l=0
        k=df['species'][i]
        c=1
    if i==len(df)-1:
        print(k,c)
        print(l//c)
    if df['petal_length'][i]>5:
        print(df.iloc[i])

    