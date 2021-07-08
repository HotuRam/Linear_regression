## ML project
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
df=pd.read_csv('linear_regression\Ecommerce Customers.csv')
df.head()
df.describe()
df.info()
sns.jointplot(x='Time on Website', y='Time on App' ,data=df)
sns.jointplot(x='Time on App', y='Yearly Amount Spent' ,data=df)
sns.pairplot(df)
df.columns
y=df['Yearly Amount Spent']
X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
ln=LinearRegression()
ln.fit(X_train,y_train)
print('Ciefficient: \n',ln.coef_)
predictions = ln.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y_test,predictions))
print('MSE',metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
coefficent= pd.DataFrame(ln.coef_,X.columns)
coefficent.columns=['Cofficients']
coefficent

