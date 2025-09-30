#QUBO-Feature-Selection

We imported the MiniBooNE particle physics dataset "X" which as 50 features and roughly 130k rows of data. 
We also have an array "y" of target values which is basically our answer key to see if feature selection went correctly (True means that the corresponding particle was a partice and False means that the data is a result of background noise)

Main task: To develop a QUBO for use in sumulated annealing to solve the issue of feature selection and compare the accuracy and speed of results with 3 classical algorithms: PCA, t-SNE and UMAP.

Important Notes so far:
- Features contain "-999". This is a placeholder for missing data.
- Features scale widely from 0.18 to 4707.
Ex: Feature 2 has values from -999 (basically N/A) to 4747 while Feature 4 has values from -999 (N/A) to 0.18.
Need to standardize this (probably use z scores or smth)