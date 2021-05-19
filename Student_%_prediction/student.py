import pandas as pd  
import numpy as np    
import matplotlib.pyplot as plt 


#read data
data_load = pd.read_excel("C:\\Users\\Rhythm\\OneDrive\\Desktop\\Student_%_prediction\\rawdata.xlsx", engine='openpyxl')  
print("\nSuccessfully imported data into console\n" )  

data_load.head(6)  


# Plot the distribution score on 2-D graph
data_load.plot(x='Hours', y='Score', style='o')    
plt.title('Hours vs Percentage')    
plt.xlabel('The Hours Studied')    
plt.ylabel('The Percentage Score')    
plt.show()  


X = data_load.iloc[:, :-1].values    
y = data_load.iloc[:, 1].values  


from sklearn.model_selection import train_test_split    
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)   


from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
print("\nTraining  Completed !.") 


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_  


print("\n")
#plotting for test data
plt.scatter(X, y)  
plt.plot(X, line);  
plt.show()  
print(X_test)   #testing data - in hours
y_pred = regressor.predict(X_test) #predicting the score
 

print("\n")
#comparing actual vs predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
print(df)
hours = [[7]]  
own_pred = regressor.predict(hours)  


print("\n Number of hours = {}\n".format(hours))   
print("\n Prediction Score = {}\n".format(own_pred[0]))  


from sklearn import metrics  
#printing the Mean Absolute Error
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 

                     
#............................................................THANK YOU !!! ..................................................................