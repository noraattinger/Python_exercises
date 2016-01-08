# Working with Lists
shopping_list = [ 'eggs', 'milk', 'cheese', 'celery', 'peanut butter', 'baking soda' ]
print(shopping_list)
print(shopping_list[2])
shopping_list[3] = 'lettuce'
shopping_list.append('chocolate bar')

# If then else...
age = 13
if age == 12:
    print('Hello')
elif age == 13:
    print('you are 13')
else:
    print('Goodbye')
    
    
# Import CSV File (Training File for Titanic Competition) as a dataframe (using Pandas)
import pandas as pd
import numpy as np
titanic = pd.read_csv('C:/Users/c125946/Desktop/Training/data/train.csv') 
titanic.describe()
# Some variables are strings

# Percentage of survived passengers
titanic["Survived"].mean()


# Exploring data
# Age distribution
import matplotlib as plt
import matplotlib.pyplot
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(titanic['Age'], bins = 10, range = (titanic['Age'].min(),titanic['Age'].max()))
plt.pyplot.title('Age distribution')
plt.pyplot.xlabel('Age')
plt.pyplot.ylabel('Count of Passengers')
plt.pyplot.show()


# Fare distribution
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(titanic['Fare'], bins = 10, range = (titanic['Fare'].min(),titanic['Fare'].max()))
plt.pyplot.title('Fare distribution')
plt.pyplot.xlabel('Fare')
plt.pyplot.ylabel('Count of Passengers')
plt.pyplot.show()

# Correlation between Pclass and Fare
titanic.boxplot(column='Fare', by = 'Pclass')

# Probability of surviving per class
temp1 = titanic.groupby('Pclass').Survived.count()
temp2 = titanic.groupby('Pclass').Survived.sum()/titanic.groupby('Pclass').Survived.count()
fig = plt.pyplot.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by class")
# Probability decreases with lower class (1st class > 60 %, 3rd class < 25 %)

# Correlation between Class, Sex, Survived
temp3 = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

# Correlation between Class, Sex, Embarkment, Survived
temp3 = pd.crosstab([titanic.Pclass, titanic.Sex, titanic.Embarked], titanic.Survived.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)



# Data preparation
# How to handle missing values: Fill with median of age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# converting sex attribute to a numeric attribute
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"]=="female","Sex"]=1

# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())

# Fill missings with most common value
titanic["Embarked"].value_counts()
# S    644
# C    168
# Q     77
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# converting Embarked attribute to a numeric attribute
titanic.loc[titanic["Embarked"] == "S", "Embarked"] =0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2