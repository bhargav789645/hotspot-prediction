from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/Classroom/Model/crime data')
#data
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/Classroom/Model/crime data')
#data
zero_counts = data.eq(0).sum(axis=0)
zero_details = data.loc[:, zero_counts > 0]
print("Features with 0 values:")
print(zero_details.columns.tolist())
print("\nNumber of 0 values in each feature:")
print(zero_counts[zero_counts > 0])
df = data.drop(['ATTEMPT TO MURDER','CULPABLE HOMICIDE NOT AMOUNTING TO MURDER','OTHER RAPE','CUSTODIAL RAPE','KIDNAPPING AND ABDUCTION OF OTHERS','PREPARATION AND ASSEMBLY FOR DACOITY','AUTO THEFT','OTHER THEFT','CRIMINAL BREACH OF TRUST','COUNTERFIETING','INSULT TO MODESTY OF WOMEN','IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES','TOTAL IPC CRIMES'],axis=1,inplace=True)
zero_counts = data.eq(0).sum(axis=0)
zero_details = data.loc[:, zero_counts > 0]
print("Features with 0 values:")
print(zero_details.columns.tolist())
print("\nNumber of 0 values in each feature:")
print(zero_counts[zero_counts > 0])
df = data.drop(['CRUELTY BY HUSBAND OR HIS RELATIVES','CAUSING DEATH BY NEGLIGENCE','DOWRY DEATHS','RIOTS','KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS'],axis=1,inplace=True)
data[data['DISTRICT']=='Total'].shape
data = data.drop(data.loc[data['DISTRICT'] == 'Total'].index)
districts = data['DISTRICT'].unique()
for district in districts:
  print(district)
data2 = data.copy()
district_encoding = {}
for i, district in enumerate(districts):
  district_encoding[district] = i

data2['DISTRICT_CODE'] = data2['DISTRICT'].map(district_encoding)
data2.head()
data2[data2['DISTRICT_CODE']==0]
data2['TOTAL IPC CASES'] = data2[['MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION', 'DACOITY', 'ROBBERY', 'BURGLARY', 'THEFT', 'CHEATING', 'ARSON', 'HURT/GREVIOUS HURT', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY', 'OTHER IPC CRIMES']].sum(axis=1)
data2['Mean_of_Crime_Values'] = data2[['MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION', 'DACOITY', 'ROBBERY', 'BURGLARY', 'THEFT', 'CHEATING', 'ARSON', 'HURT/GREVIOUS HURT', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY', 'OTHER IPC CRIMES']].mean(axis=1)
data3 = data2.copy()
data4 = data3.copy()
df1 = data4.iloc[:,1:2]
data5 = data4.copy()
data5 = pd.get_dummies(data4, columns=['DISTRICT'])
data6 = pd.concat([data5, df1], axis=1)
# prompt: apply the randomforest algorithm on the dataset to find the crime hotspot place
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Separate features and target
X = data6.drop(['STATE/UT','DACOITY', 'YEAR','ROBBERY', 'BURGLARY',  'ARSON', 'HURT/GREVIOUS HURT', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY', 'OTHER IPC CRIMES','DISTRICT'], axis=1)
y = data6['DISTRICT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rfc = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Find the most important features
importances = rfc.feature_importances_
features = X.columns
for i, feature in enumerate(features):
  print(f"{feature}: {importances[i]:.4f}")

# Identify the crime hotspot place
hotspot_index = np.argmin(y_pred)
hotspot_district = data4.iloc[hotspot_index]['DISTRICT']
print(f"Crime hotspot place: {hotspot_district}")
A= data6['MURDER'].mean()
B = data6['RAPE'].mean()
C=data6['KIDNAPPING & ABDUCTION'].mean()
D=data6['THEFT'].mean()
E =data6['CHEATING'].mean()
print(A)
print(B)
print(C)
print(D)
print(E)
average = (A+B+C+D+E) /5
print((average*14)/5)
# prompt: give me top 10 crime hotspots based on above accuracy

hotspot_indices = np.argsort(y_pred)[::-1][:10]
hotspot_districts = data4.iloc[hotspot_indices]['DISTRICT'].tolist()

print("Top 10 crime hotspots:")
for district in hotspot_districts:
  print(f"\t{district}")
# prompt: take these districts on x -axis and total ipc crimes on y-axis

import matplotlib.pyplot as plt

# Prepare data
districts = hotspot_districts
total_ipc_crimes = []
for district in districts:
  total_ipc_crimes.append(data4[data4['DISTRICT'] == district]['TOTAL IPC CASES'].values[0])

# Create bar chart
plt.bar(districts, total_ipc_crimes)
plt.xlabel("Districts")
plt.ylabel("Total IPC Crimes")
plt.title("Top 10 Crime Hotspots in India (2001-2012)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
data6[data['DISTRICT']=='GAUTAMBUDH NAGAR']['Mean_of_Crime_Values']
red = []
normal = []

# Convert the relevant column to boolean
data6.loc[data6['YEAR'] == 2014, 'Mean_of_Crime_Values'] = data6.loc[data6['YEAR'] == 2014, 'Mean_of_Crime_Values'].astype(bool)

# Iterate over rows of data6
for index, row in data6.iterrows():
    if row['Mean_of_Crime_Values'] > average:
        red.append(row['DISTRICT'])
    else:
        normal.append(row['DISTRICT'])

print(int(len(red)))
print(red)

