import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
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
# holding original, easier to interpret the words than when they are converted to numbers
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name', 'ticket', 'home.dest'], 1, inplace=True)
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# new group
original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]  # for column / group -> put in labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    # temp_df is original_df only where cluster group is ex. cluster group 0 (or 1, or 2...)
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    # again survival cluster is the temp_df for the people that survived
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    # good for comparing survival rate between clusters
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
