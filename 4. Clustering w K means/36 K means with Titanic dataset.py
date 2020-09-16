import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import xdrlib
style.use('ggplot')

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Abroad
parch Number of Parents/Children Abroad
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)  # non descriptive features
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}  # mapping of text to number conversion

        def convert_to_int(val):  # once we have the mapping we apply this to all non-numeric data
            return text_digit_vals[val]

        # is the column a number?
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)  # so we don't iterate through everything
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:  # if not already given an index then give a number to it
                    text_digit_vals[unique] = x
                    x += 1

            # converting the column
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)
print('Changed non-numerical data to numerical data:')
# print(df.head())

# df.drop(['ticket'], 1, inplace=True)  # doesn't seem to make much of a difference
# df.drop(['boat'], 1, inplace=True)  # seems to drop accuracy just slightly

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)  # don't know what this does but makes a huge change in the final value
y = np.array(df['survived'])  # only used to check if k means worked

clf = KMeans(n_clusters=2)
clf.fit(X)

# difference between supervised learning and unsupervised learning
# here we can ask it to predict each X value as it predicts based on the location of each centroid
# in supervised learning it would already know the answer
# ALSO we haven't even told it to look for survivors
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)  # based on centroid
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))  # 20% here would be 80% (80 would be 80, 10 would be 90, 5 would be 95...)