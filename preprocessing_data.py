"""
reading data from csv
"""
import pandas as pd
data = pd.read_csv("Data.csv")

"""
seperating independent and dependent data"
"""
independent = data.iloc[:,:3].values
dependent = data.iloc[:,3].values

"""
filling in missing values
"""
from sklearn.preprocessing import Imputer
i = Imputer()
independent[:,1:3] = i.fit_transform(independent[:,1:3])


"""
Categorizing data
"""
from sklearn.preprocessing import LabelEncoder
l =LabelEncoder()#assigns numbers to strings
independent[:,0] = l.fit_transform(independent[:,0])
dependent = l.fit_transform(dependent)

"""
encoding the categorized data
"""
from sklearn.preprocessing import OneHotEncoder
a = OneHotEncoder(categorical_features = [0])
independent = a.fit_transform(independent).toarray()


"""
seperating test set and train set
"""
from sklearn.model_selection import train_test_split 
i_train,i_test,d_train,d_test = train_test_split(independent,dependent,
                                                 test_size = 0.2,
                                                 random_state = 42)

"""
Feature Scaling
"""
#this is being done due to increased eucledian distane bw age and salary
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
i_train = s.fit_transform(i_train)
i_test = s.transform(i_test)
