# # Análisis Exploratorio de Datos (EDA)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image
sns.set()

# # Pandas

test = pd.DataFrame(np.array([1,2,3]))

df = pd.read_csv('recent-grads.csv')

# # Resumen estadístico y examinación de variables
df.shape

df.head()

df.info()

df.describe()

df.isna().sum()

df.dropna(inplace = True)

df.isna().sum()

df['Men'].describe()

df.Major_category.unique().shape

df.Men.describe()

# # Gráfico básico con Matplotlib

plt.scatter(df['Men'], df['Employed'])
plt.xlabel('Test 1')

# # Seaborn

fig = plt.figure(figsize = (25, 10))
sns.barplot(x = 'Major_category', y = 'Men', data = df)
plt.xticks(rotation = 90)
plt.xlabel('Hola')
plt.title('Major Category x Men', size = 20)


# # Análisis de 1 variable

plt.figure(figsize = (25,10))
ax = sns.distplot(df['Men'])

print('Asimetría: {} \nCurtosis: {}'.format(df['Men'].skew(), df['Men'].kurt()))

df['Men'].describe()

plt.figure(figsize = (25,10))
ax = sns.distplot(df['Women'])

# # Análisis de 2 variables

# # Barplots

df[df['Major_category'] == 'Engineering'].head()

fig = plt.figure(figsize = (25, 10))
sns.barplot(x = 'Major_category', y = 'Women', data = df)
plt.xticks(rotation = 90)


plt.figure(figsize = (25, 10))
ax = sns.barplot(x = 'Major', y = 'Men', data = df[df['Major_category'].isin(['Engineering', 'Business'])], hue = 'Major_category')
plt.xticks(rotation = 90)


plt.figure(figsize = (25, 10))
ax = sns.barplot(x = 'Major', y = 'Men', data = df[(df['Major_category'] == 'Engineering') & (df['Unemployed'] < 300)])
plt.xticks(rotation = 90)

# # Boxplots

plt.figure(figsize = (25,10))
ax = sns.boxplot(df['Men'])
plt.xticks(rotation = 90, size = 20)

df['Men'].describe()

plt.figure(figsize = (25,10))
ax = sns.boxplot(x = 'Major_category', y = 'Men', data = df)
plt.xticks(rotation = 90)

plt.figure(figsize = (25,10))
ax = sns.boxplot(df['Women'])
plt.xticks(rotation = 90)

df['Women'].describe()


df['Women'].quantile(0.75) - df['Women'].quantile(0.25)

plt.figure(figsize = (25,10))
ax = sns.boxplot(x = 'Major_category', y = 'Women', data = df)
plt.xticks(rotation = 90)


# # Scatter Plots

plt.figure(figsize = (25,10))
ax = sns.regplot(x = "Men", y="Unemployed", data = df)

df[['Men', 'Unemployed']].corr()


plt.figure(figsize = (25,10))
ax = sns.regplot(x = "Rank", y="P75th", data = df)


plt.figure(figsize = (25,10))
ax = sns.regplot(x = "Rank", y="Median", data = df) 


plt.figure(figsize = (25,10))
ax = sns.regplot(x = "Median", y="ShareWomen", data = df) 

plt.figure(figsize = (25,10))
ax = sns.regplot(x = "Median", y="Unemployment_rate", data = df) 

df[['Median', 'Unemployment_rate']].corr()

# # Correlación de variables

corr = df.drop(['Major', 'Major_category'], axis = 1).corr()


plt.figure(figsize = (20, 20))
ax = sns.heatmap(corr.round(1) * 100, annot=True, fmt=".0f")

# # Pair Plot

sns.pairplot(df.drop(['Major', 'Major_category'], axis = 1))
plt.xticks(size = 15)

