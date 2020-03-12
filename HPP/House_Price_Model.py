#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

#loading a datasets
df = pd.read_csv("C:\\Users\\Administrator\\Downloads\\House_Price_Prediction\\House_Price_Prediction\\HPP\\Bengaluru_House_Data.csv")
#print(df.head())

df['area_type'].unique()

df['area_type'].value_counts()

#Drop features that are not required to build our model
df.drop(['area_type','society','balcony','availability'],axis='columns',inplace=True)

#Checking for null values
df.isna().sum()

#-------------------Data Cleaning-----------------

#Handle NA values : Remove all the rows which contain the NA value
df1 = df.dropna()

#Again checking for null values
df1.isna().sum()

#------------------Feature Engineering---------------

df1['size'].unique()

#Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
df1['bhk'] = df1['size'].apply(lambda x: int(x.split(' ')[0]))

df1['bhk'].unique()
df1[df1.bhk>20]

df1['total_sqft'].unique()

#Explore total_sqft feature
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True            


df1[~df1['total_sqft'].apply(is_float)].head(10)

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) ==2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

convert_sqft_to_num('1015 - 1540')                      #return the average value

convert_sqft_to_num('2323')                           #return the simple float value

convert_sqft_to_num('34.46Sq. Meter')                  #return nothing because these type of rows will be removed

df2 = df1.copy()
df2['total_sqft'] = df2['total_sqft'].apply(convert_sqft_to_num)
#print(df2.head())

#For below row, it shows total_sqft as 2475 which is an average of the range 2100-2850
df2.loc[30]

#---------------Feature Engineering----------------

#Add new feature called price per square feet
df3 = df2.copy()
df3['price_per_sqft'] = df3['price']*100000/df3['total_sqft']
#print(df3.head())

df3_stats = df3['price_per_sqft'].describe()

df3.location.unique()
len(df3.location.unique())

#Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations
df3.location = df3.location.apply(lambda x: x.strip())
location_stats = df3['location'].value_counts(ascending=False)

location_stats.values.sum()

len(location_stats[location_stats>10])

len(location_stats)

len(location_stats[location_stats<10])

#-----------------Dimensionality Reduction----------------

location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

df3['location'].unique()

len(df3['location'].unique())

df3['location'] = df3['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df3['location'].unique())

#print(df3.head(10))

# Outlier Removal Using Business Logic
df3[df3['total_sqft']/df3.bhk<300].head()

'''Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. 
These are clear data errors that can be removed safely'''

df4 = df3[~(df3.total_sqft/df3.bhk<300)]

# Outlier Removal Using Standard Deviation and Mean
df4.price_per_sqft.describe()

'''Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices.
We should remove outliers per location using mean and one standard deviation'''

def remove_pps_outliers(df_new):
    df_out = pd.DataFrame()
    for key, subdf in df_new.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df4 = remove_pps_outliers(df4)

#print(df4)

def plot_scatter_chart(df_new,location):
    bhk2 = df_new[(df_new.location==location) & (df_new.bhk==2)]
    bhk3 = df_new[(df_new.location==location) & (df_new.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price,marker='+', color='green',label='3 BHK',s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df4,"Rajaji Nagar")
#plt.show()

#Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

def remove_bhk_outliers(df_new):
    exclude_indices = np.array([])
    for location, location_df in df_new.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df_new.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)

#print(df5)

#Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties
plot_scatter_chart(df5,"Rajaji Nagar")
plot_scatter_chart(df5,"Hebbal")

matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df5.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
#plt.show()

# Outlier Removal Using Bathrooms Feature

df5['bath'].unique()

plt.hist(df5.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

df5[df5['bath']>10]

df5[df5['bath']>df5['bhk']+2]

df6 = df5[df5.bath<df5.bhk+2]

df7 = df6.drop(['size','price_per_sqft'],axis='columns')
#print(df7.head(3))

#Use One Hot Encoding For Location

ohe = pd.get_dummies(df7.location)

df8 = pd.concat([df7,ohe.drop('other',axis='columns')],axis='columns')

df9 = df8.drop('location',axis='columns')

#seperate dependent and independent variable from datasets

X = df9.drop(['price'],axis='columns')
y = df9['price']

len(y)

#Splitting the dataset into Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#Creating/Fitting a Simple Linear Regression model

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

#KFold cross validation

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

# Find best model using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)

np.where(X.columns=='2nd Phase Judicial Layout')[0][0]             #return the column index

# Test the model for few properties
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

#Output
j = predict_price('1st Phase JP Nagar',1000, 2, 2)
print(j)
