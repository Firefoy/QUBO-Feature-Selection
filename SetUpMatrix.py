from sklearn.datasets import fetch_openml

data = fetch_openml(name='MiniBooNE', version=1, parser='auto')

X = data.data 
y = data.target  
X = X.values
y = y.values

# At this point the dataset is loaded and ready to work with
# X is 130,064 samples with 50 features
# y is the target variable and its basically an array of TRUE and FALSE