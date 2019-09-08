#This code uses machine learnign algorithms to do population forecasting
#for a cohort of ACT residents: by gender, age and suburb.

import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class Population_Forecast:
    def __init__(self, datain, year0, baseyr):
        self.datain = datain
        self.year0 = year0
        self.baseyr = baseyr
        self.df = pd.read_csv(self.datain)
        sexv = self.df['Sex'].unique().tolist()
        self.sex_map = dict( zip(sexv,[-1,1]) )
        agev = self.df['Age Group'].unique().tolist()
        self.age_map = dict( zip(agev,range(1,len(agev)+1)) )
        self.df['Suburb'] = self.df['Suburb'].str.strip()
        
    def data_trans(self):
        self.df['Year'] = pd.DatetimeIndex(self.df['Year']).year
        #Calculate population percentage
        year_tot = self.df.groupby(['Year']).agg({'Population': 'sum'})
#        print(year_tot)
        df2 = pd.merge(self.df, year_tot, on='Year', how='left')
        #print(df2)
        df2['pect'] = df2['Population_x'].div(df2['Population_y'], axis=0)*100
        
        #transform data
        df2['Yearn'] = df2['Year'] - self.year0
        #print(df2)
        
#        print(sex_map)
        df2.replace({'Sex': self.sex_map},inplace=True)
        #print(df2)
#        print(age_map)
        df2.replace({'Age Group': self.age_map},inplace=True)
#        print(df2)
        self.df2a = df2
        sub_encode = pd.get_dummies(df2['Suburb'], prefix = 'Suburb')
        df2 = pd.concat([df2, sub_encode], axis=1)
        df2 = df2.drop(['Year','Suburb','Population_x','Population_y'], axis=1)

        self.df2 = df2
        return self.df2, self.df2a

    def model(self, dft, result=True):
        y = dft['pect']
        x = dft.drop(['pect'], axis=1)
#        print(y.shape)
#        print(x.shape)
        
        indices = np.array(range(dft.count()[0]))
        X_train, X_test, y_train, y_test, indices_train,indices_test = train_test_split(x, y, indices, random_state=1)
#        print(X_train.shape)
#        print(X_test.shape)
#        print(y_train.shape)
#        print(y_test.shape)
#        print('transformed :', X_test.columns)
        self.regressor = LinearRegression()  
#        self.regressor = SVR(kernel='poly', C=100, degree=2)
#        self.regressor = DecisionTreeRegressor(random_state = 100) 
        self.regressor.fit(X_train, y_train)
        y_pred = self.regressor.predict(X_test)

        if result == True:
            #To retrieve the intercept:
            print('model intercept: ', self.regressor.intercept_)
            #For retrieving the slope:
            print('model coefficient: ', self.regressor.coef_)
            cmp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            print(cmp)
            
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
            print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
            print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        
        else:
            pass
#        return model
        return self.regressor
    
    
    def predict(self, X_list):
        age = X_list[3]
        if 0 <= age <= 10:
            age_grp = '0-10'
#            age_grpv = 1
        elif 11 <= age <= 20:
            age_grp = '11-20'
        elif 21 <= age <= 30:
            age_grp = '21-30'
        elif 31 <= age <= 40:
            age_grp = '31-40'            
        elif 41 <= age <= 50:
            age_grp = '41-50'
        elif 51 <= age <= 60:
            age_grp = '51-60'
        elif age > 60:
            age_grp = '60+'

#        print(age_grp)
        X_dict = {'Year': X_list[0], 'Suburb': X_list[1], 'Sex': X_list[2], 'Age Group': age_grp}
        X_df = pd.DataFrame(X_dict, index=[0])
#        print('User input is: ', X_df)

        dfcopy = self.df.drop(['Population'], axis=1)
#        print(dfcopy)
        dfn = dfcopy.append(X_df)
#        print(dfn)
#        return dfn
                
        #transform data
        dfn['Yearn'] = dfn['Year'] - self.year0
        #print(df2)

#        print(sex_map)
        dfn.replace({'Sex': self.sex_map},inplace=True)
        #print(df2)
#        print(age_map)
        dfn.replace({'Age Group': self.age_map},inplace=True)
#        print(df2)
        sub_encoden = pd.get_dummies(dfn['Suburb'], prefix = 'Suburb')
        dfn = pd.concat([dfn, sub_encoden], axis=1)
        dfn = dfn.drop(['Year','Suburb'], axis=1)
        X_val = dfn.iloc[-1,:]
        print('Predict for: ', X_df)
#        print(dfn.iloc[-1,:])
#        print(list(dfn.columns))
        y_result = float(self.regressor.predict(dfn)[-1])
#        print(list(X_df2.columns))
#        print(X_test2)
#        Calculate population growth rate compared to base year
        base = dfcopy.loc[(dfcopy['Year'] == self.baseyr) & (dfcopy['Suburb'] == X_list[1]) & (dfcopy['Sex'] == X_list[2]) & (dfcopy['Age Group'] == age_grp)]
        print(base)
        base_ind = base.index[0]
#        print('base row index: ', base_ind)
#        print('basee transformed: ', self.df2a.iloc[base_ind])
        pect0 = float(self.df2a.iloc[base_ind]['pect'])
        print('base population %: ', pect0)
        pgr = (y_result - pect0)/pect0
        return y_result, pgr
       
#Test model (Forecasting)
Pop_Fcst = Population_Forecast(r'C:\Users\HjAcer\Documents\Govhack\Govhack 2019\ACT_Pop_Projct_Suburb_T.csv',2015,2019)
data_trans, data_transa = Pop_Fcst.data_trans()
pop_fcst_model = Pop_Fcst.model(data_trans, result=False)

result = []
grwothrate = []
for i in range(2019,2040):
#    user_input = [i, 'Mawson', 'FEMALE', 35]
    user_input = [i, 'Harrison', 'FEMALE', 35]
#    user_input = [i, 'Gungahlin East', 'MALE', 35]
    resulti, grwothratei = Pop_Fcst.predict(user_input)
    print('Prediction result for year ',i,'is: ', resulti)
    print('Population growth rate (compared to ',2019,') is: ',grwothratei)
    grwothrate.append(grwothratei)
    result.append(resulti)
    
grwothzip = list(zip(range(2019,2040), grwothrate))
for i in range(len(grwothzip)):
    print(grwothzip[i])
    
resultzip = list(zip(range(2019,2040), result))
for i in range(len(resultzip)):
    print(resultzip[i])
