<H3>ENTER YOUR NAME: VIGNESH V</H3>
<H3>ENTER YOUR REGISTER NO:212224230303</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 3/05/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import drive
drive.mount('/content/drive')

#import libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
data = pd.read_csv("/content/drive/MyDrive/50_Startups.csv")
print(data.head())
print(data.tail())

# Finding Missing Values
print("Missing Values: \n ",data.isnull().sum())

#Handling Missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

#Check for Duplicates
print("Duplicate values:\n ")
print(data.duplicated())

#Detect Outliers
data.describe()

#Normalize the dataset
scaler=MinMaxScaler()
# Select only numerical columns for scaling
numeric_cols = data.select_dtypes(include=['number']).columns
df1 = pd.DataFrame(scaler.fit_transform(data[numeric_cols]), columns=numeric_cols)
print("Normalized data \n" , df1)

#split the dataset into input and output
X = data.drop('Profit', axis=1)
y = data['Profit']

#splitting the data for training & Testing
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Print the training data and testing data
print("Training data")
print(X_train)
print(y_train)

print("Testing data")
print(X_test)
print(y_test)
print("Length of X_test: ", len(X_test))
```
## OUTPUT:

**Read the dataset from drive:**
<img width="594" height="217" alt="image" src="https://github.com/user-attachments/assets/3221a140-07e2-4a41-a34f-73f46f4d4c89" />

**Finding Missing Values:**
<img width="303" height="137" alt="image" src="https://github.com/user-attachments/assets/533edfaa-9dcf-48ec-a344-4b776f7d8a0e" />

**Check for Duplicates:**
<img width="175" height="781" alt="image" src="https://github.com/user-attachments/assets/50ecbe61-a3ae-4f3e-8913-2fe5e372eada" />

**Detect Outliers:**
<img width="485" height="254" alt="image" src="https://github.com/user-attachments/assets/361e07cf-12da-4be9-90ed-bfd5800e464e" />

**Normalize the dataset:**
<img width="447" height="765" alt="image" src="https://github.com/user-attachments/assets/5e2d3294-63fe-464f-a0a8-cef354184c5f" />

**Print the training data and testing data:**
<img width="526" height="846" alt="image" src="https://github.com/user-attachments/assets/c26620d8-9c96-4f87-ac3c-59a13c3d3ce0" />
<img width="500" height="857" alt="image" src="https://github.com/user-attachments/assets/67836809-c2c2-4c96-8c74-5f0bc1ceec7e" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


