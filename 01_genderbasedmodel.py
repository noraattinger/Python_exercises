# Women are more likely to survive
# MODEL 1: Prediction according to gender (all women survive, all men die)

# The first thing to do is to import the relevant packages
# that I will need for my script, 
# these include the Numpy (for maths and arrays)
# and csv for reading and writing csv files
# If i want to use something from this I need to call 
# csv.[function] or np.[function] first

import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
#csv_file_object = csv.reader(open('../csv/train.csv', 'rb')) 
csv_file_object = csv.reader(open('C:/Users/c125946/Desktop/Training/data/train.csv', 'rb')) 

header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	          # Then convert from a list to an array
        			          # Be aware that each item is currently
                                 # a string in this format

# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

number_passengers = np.size(data[0::,1].astype(np.float))#891
number_survived = np.sum(data[0::,1].astype(np.float))#342
proportion_survivors = number_survived / number_passengers#38%


women_only_stats = data[0::,4] == "female" # This finds where all 
                                           # the elements in the gender
                                           # column that equals “female”
men_only_stats = data[0::,4] != "female"   # This finds where all the 
                                           # elements do not equal 
                                           # female (i.e. male)


# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard) 

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
#Proportion of women who survived is 0.742038216561
print 'Proportion of men who survived is %s' % proportion_men_survived
#Proportion of men who survived is 0.188908145581



### Reading the test data and writing the gender model as a csv
test_file = open('C:/Users/c125946/Desktop/Training/data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
prediction_file = open("C:/Users/c125946/Desktop/Training/data/genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()