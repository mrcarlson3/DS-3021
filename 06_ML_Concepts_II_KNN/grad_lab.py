"""
Graduation Lab: Week 6

Instructions:

Let's build a kNN model using the college completion data. 
The data is messy and you have a degrees of freedom problem, as in, we have too many features.  

You've done most of the hard work already, so you should be ready to move forward with building your model. 

1. Use the question/target variable you submitted and 
build a model to answer the question you created for this dataset (make sure it is a classification problem, convert if necessary). 

2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning
if needed changed the target variable.

3. Create a dataframe that includes the test target values, test predicted values, 
and test probabilities of the positive class.

4. No code question: If you adjusted the k hyperparameter what do you think would
happen to the threshold function? Would the confusion look the same at the same threshold 
levels or not? Why or why not?

5. Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what 
concerns or positive elements do you have about the model as it relates to your question? 

6. Create two functions: One that cleans the data & splits into training|test and one that 
allows you to train and test the model with different k and threshold values, then use them to 
optimize your model (test your model with several k and threshold combinations). Try not to use variable names 
in the functions, but if you need to that's fine. (If you can't get the k function and threshold function to work in one
function just run them separately.) 

7. How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

8. Choose another variable as the target in the dataset and create another kNN model using the two functions you created in
step 7. 

"""
# README for the dataset - https://data.world/databeats/college-completion/workspace/file?filename=README.txt

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import metrics
from numpy import log

grad_data = pd.read_csv('https://query.data.world/s/qpi2ltkz23yp2fcaz4jmlrskjx5qnp', encoding="cp1252")
# the encoding part here is important to properly read the data!

#%%
# We have a lot of data! A lot of these have many missing values or are otherwise not useful.
to_drop = list(range(39, 56))
to_drop.extend([27, 9, 10, 11, 28, 36, 60, 56])
#%%
grad_data1 = grad_data.drop(grad_data.columns[to_drop], axis=1)
#%%
# drop even more data that doesn't look predictive
drop_more = [0,2,3,6,8,11,12,14,15,18,21,23,29,32,33,34,35]
grad_data2 = grad_data1.drop(grad_data1.columns[drop_more], axis=1)
grad_data2.info()
#%%
grad_data2.replace('NULL', np.nan, inplace=True)
#%%
grad_data2['hbcu'] = [1 if grad_data2['hbcu'][i]=='X' else 0 for i in range(len(grad_data2['hbcu']))]
grad_data2['hbcu'].value_counts()
grad_data2['hbcu'] = grad_data2.hbcu.astype('category')
# convert more variables to factors
grad_data2[['level', 'control']] = grad_data2[['level', 'control']].astype('category')

#%%
# check missing data
sns.displot(
    data=grad_data2.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25
)
#%%
#let's drop med_stat_value then delete the rest of the NA rows
grad_data2 = grad_data2.drop(grad_data[['med_sat_value']], axis=1)
grad_data2.dropna(axis = 0, how = 'any', inplace = True)

#%%
#Start of assignment
#Q1: build a model to answer the question you created for this dataset
#dataset for assignment
df = grad_data2.drop(['chronname'], axis=1)
df['aid_value'] = df['aid_value'].replace(0, np.nan)
df['log_aid'] = np.log(df['aid_value'])
df['aid'] = pd.cut(df['log_aid'], bins=[0, 5.5, 8.6, 10.7], labels=['low', 'medium', 'high'])

#Q1 Question: Can we classify aid value based on the other features of the dataset?
#%%
def clean_and_split_data(df, target, test_size=0.4, val_size=0.5, random_state=1984):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = preprocessing.MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    cat_cols = ['level', 'control', 'hbcu']
    df[cat_cols] = df[cat_cols].astype('category')
    df = pd.get_dummies(df, columns=cat_cols)
    
    # One-hot encode 'aid'
    df = pd.get_dummies(df, columns=['aid'], prefix='aid')
    
    # Split data into train, test, and validation sets
    train, test = train_test_split(df, test_size=test_size, stratify=df['aid_low'], random_state=random_state)
    test, val = train_test_split(test, test_size=val_size, stratify=test['aid_low'], random_state=random_state)
    
    return train, test, val

#%%
# Clean and split data
train, test, val = clean_and_split_data(df, 'aid')

# Prepare features and labels
X_train = train.drop(['aid_low', 'aid_medium', 'aid_high'], axis=1)
X_val = val.drop(['aid_low', 'aid_medium', 'aid_high'], axis=1)
X_test = test.drop(['aid_low', 'aid_medium', 'aid_high'], axis=1)

y_train = train[['aid_low', 'aid_medium', 'aid_high']].idxmax(axis=1)
y_val = val[['aid_low', 'aid_medium', 'aid_high']].idxmax(axis=1)
y_test = test[['aid_low', 'aid_medium', 'aid_high']].idxmax(axis=1)

#%%

#Q2. Build a kNN model to predict your target variable using 3 nearest neighbors. Make sure it is a classification problem, meaning if needed changed the target variable.
# Train KNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#%%
# Evaluate model
train_accuracy = neigh.score(X_train, y_train)
val_accuracy = neigh.score(X_val, y_val)
test_accuracy = neigh.score(X_test, y_test)

#%%
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

y_test_pred = neigh.predict(X_test)
y_test_prob = neigh.predict_proba(X_test)


#%%
#Q3. Create a dataframe that includes the test target values, test predicted values, and test probabilities of the positive class.
y_val_pred = neigh.predict(X_val)
results = pd.DataFrame({
    'Actual': y_test.tolist(),
    'Predicted': y_test_pred.tolist(),
    'Probabilities': y_test_prob.tolist()
})

print(results.head())

#%%
#Q4. If you adjusted the k hyperparameter what do you think would happen to the threshold function? Would the confusion look the same at the same threshold levels or not? Why or why not?
# If we adjusted the k hyperparameter, the threshold function would change. A smaller k value would make the model more sensitive to noise in the data, potentially leading to more false positives and negatives as well as overfitting the data showing false accuracy .
# A larger k value would smooth out the predictions, making the model less sensitive to noise but potentially missing some of the finer details in the data, leading to underfitting. 
# As a result, the confusion matrix looks different at different k values, as the model's performance would vary with changes in k as it begins to underfit the data more and more. For my data, it looks like the best accuracy is at k=7 and begins to drop after this point. 

#%%
#Q5: Evaluate the results using the confusion matrix. Then "walk" through your question, summarize what concerns or positive elements do you have about the model as it relates to your question?
# Generate confusion matrix
cm = confusion_matrix(y_val, y_val_pred, labels=neigh.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)
disp.plot()
plt.show()
# The concerns I have about the model is that the accuracy stays around 83% for the test data, while the training data is much higher. This could mean there is overfitting and that the model would not do well on new data. The positive element is it seems that it is able to classify the data well with very 
#little false negatives or positives. It makes me wonder what features are the best to incorporate as the predictors of aid value.
#%%
# Generate classification report
print(classification_report(y_val, y_val_pred))

# Function to find optimal k
def chooseK(k, X_train, y_train, X_test, y_test):
    random.seed(1)
    class_knn = KNeighborsClassifier(n_neighbors=k)
    class_knn.fit(X_train, y_train)
    return class_knn.score(X_test, y_test)

# Test different k values
test_results = pd.DataFrame({'k': list(range(1, 22, 2)), 
                             'accu': [chooseK(i, X_train, y_train, X_test, y_test) for i in range(1, 22, 2)]})
print(test_results)

#%%
#Q6: Create two functions: One that cleans the data & splits into training|test and one that allows you to train and test the model with different k and threshold values. 
def clean_and_split(df, target, categories, test_size=0.4, val_size=0.5, random_state=1984):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = preprocessing.MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    cat_cols = categories
    df[cat_cols] = df[cat_cols].astype('category')
    df = pd.get_dummies(df, columns=cat_cols)
    
    # One-hot encode target
    df = pd.get_dummies(df, columns=[target], prefix=target)

    # Split data into train, test, and validation sets
    train, test = train_test_split(df, test_size=test_size, stratify=df[f'{target}_low'], random_state=random_state)
    test, val = train_test_split(test, test_size=val_size, stratify=test[f'{target}_low'], random_state=random_state)
    
    X1_train = train.drop([f'{target}_low', f'{target}_medium', f'{target}_high'], axis=1)
    y1_train = train[[f'{target}_low', f'{target}_medium', f'{target}_high']].idxmax(axis=1)
    
    X1_test = test.drop([f'{target}_low', f'{target}_medium', f'{target}_high'], axis=1)
    y1_test = test[[f'{target}_low', f'{target}_medium', f'{target}_high']].idxmax(axis=1)
    
    X1_val = val.drop([f'{target}_low', f'{target}_medium', f'{target}_high'], axis=1)
    y1_val = val[[f'{target}_low', f'{target}_medium', f'{target}_high']].idxmax(axis=1)
    
    return X1_train, y1_train, X1_test, y1_test, X1_val, y1_val


def choose(k, X_train, y_train, X_test, y_test, threshold):
    random.seed(1)
    class_knn = KNeighborsClassifier(n_neighbors=k)
    class_knn.fit(X_train, y_train)
    
    # Predict probabilities
    y_prob = class_knn.predict_proba(X_test)
    
    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Convert predictions to labels
    y_pred_labels = []
    for prob in y_prob:
        if max(prob) >= threshold:
            y_pred_labels.append(class_knn.classes_[np.argmax(prob)])
        else:
            y_pred_labels.append('unknown')
    
    return metrics.accuracy_score(y_test, y_pred_labels)

#%%
#Q7: How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not?
X1_train, y1_train, X1_test, y1_test, X1_val, y1_val = clean_and_split(df, 'aid', ['level', 'control', 'hbcu'])
test1 = choose(7, X1_train, y1_train, X1_test, y1_test, 0.9)

print(test1)
# The model performed on par with the previous model the adjustment of the k value only altered the model slightly by 0.01. The adjusted thresholds did not help the model as decreasing it made it the same while increasing the threshold dramatically decreased the accuracy.
#%%
#Q8: Choose another variable as the target in the dataset and create another kNN model using the two functions you created in step 7.
df['retain'] = df['retain_value'].replace(0, np.nan)
df['retain'] = pd.cut(df['retain'], bins=[0, 0.5, 0.75, 1], labels=['low', 'medium', 'high'])                                 
X2_train, y2_train, X2_test, y2_test, X2_val, y2_val = clean_and_split(df, 'retain', ['level','control', 'hbcu', 'aid'])
test2 = choose(7, X2_train, y2_train, X2_test, y2_test, 0.2)
print(test2)