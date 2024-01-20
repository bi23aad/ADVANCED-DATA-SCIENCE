#importing the libraries needed
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit

#importing the data
def read_data():
    data = pd.read_csv("Final_Dataset.csv")
    return data
data = read_data()

# Clean data by replace non-number with pd.NA
data = data.replace('..', pd.NA)

# Handle Missing Values
data = data.fillna(method='ffill').fillna(method='bfill')
data = data.drop(['Series Code', 'Time Code'], axis=1)
 
#Transpose the data
transposed = pd.melt(data,
                         id_vars=['Series Name','Time'],
                         var_name='Country Name',
                         value_name='Value')
transposed = transposed.pivot_table(
    index=['Time', 'Country Name'],
    columns='Series Name',
    values='Value',
    aggfunc='first')

# Reset index for a clean structure
transposed.reset_index(inplace=True)
transposed.iloc[:, 2:] = transposed.iloc[:, 2:].astype("float")

#Heatmap To View Correlations between Indicators
corr_matrix = transposed.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

#Select only the numerical values in the dataset
num_cols = transposed.select_dtypes(include=[np.number])
'''select indicators to be used for clustering. The indicators were
selected based on the low correlation value between them'''
try_tip = num_cols[['Methane emissions (% change from 1990)', 'Forest area (sq. km)']]

#clean the data to be clustered
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)
cluster_data = clean_dataset(try_tip)

#Normalize the data
def normalize(df):
    df_min = df.min()
    df_max = df.max()
    df = (df-df_min) / (df_max - df_min) 
    return df, df_min, df_max
scale, a, b = normalize(cluster_data)

#Carry Out the clustering using Sklearn model
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, n_init=20) 
    model = kmeans.fit(scale)
    clusters = kmeans.predict(scale) 
    silhouette = silhouette_score(scale, clusters) 
    print("Number of Clusters =", k) 
    print("Silhouette Score =", silhouette) 
    print()
    
#The cluster number with the silhouette closest to +1 was used
kmeans = KMeans(n_clusters=3).fit(scale)

labels = kmeans.labels_
cen = kmeans.cluster_centers_
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

#Visualizations of clusters
plt.figure(figsize=(10, 6))
plt.scatter(scale.iloc[:,0], scale.iloc[:,1], c=labels, cmap='rainbow')
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.title('Data Clusters Based On Methane Emission And Forest Area  Values')
plt.xlabel('Methane Emissions')
plt.ylabel('Forest Area(Sq. Km)')
plt.legend(['Cluster 1', 'Cluster 0', 'cluster 2'])
plt.colorbar(ticks=range(3))
plt.show()


#Exponential Fitting Function
ydata = transposed['CO2 emissions (metric tons per capita)']
xdata = transposed['Population growth (annual %)']

def exponential_growth(x, a, b):
    return a * b**x
popt, pcov = curve_fit(exponential_growth, xdata, ydata,)
initial_value, growth_factor = popt
y_fitted = exponential_growth( xdata, *popt)
print(initial_value)
print(growth_factor)
x_line = np.linspace(min(xdata), max(xdata), num=100)
y_line = exponential_growth(x_line, *popt)
conf_interval = np.abs(np.diag(pcov)) * 1.96
y_upper = exponential_growth(xdata, popt[0]+conf_interval[0], popt[1]+conf_interval[1])
y_lower = exponential_growth(xdata, popt[0]-conf_interval[0], popt[1]-conf_interval[1])

print(conf_interval)
plt.plot(xdata, ydata, 'o', label='data')
plt.plot(x_line, y_line, '--', label='fit')
plt.fill_between(xdata, y_lower, y_upper, alpha=0.5, label='95% CI')
plt.xlabel('Population Growth')
plt.ylabel('CO2 Emissions')
plt.title("Exponential Fitting Function of Populaation Growth against CO2 Emissions ")
plt.legend()
plt.show()

'''Prediction use the exponential curve fit'''
x_data = 6.20105
y_pred = exponential_growth(x_data, *popt) 
print(y_pred)
print("Above is the potential value for CO2 Emissions per Capita assuming the population Growth stands at 6.20105")





