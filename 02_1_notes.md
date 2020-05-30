# Chapter 2: End-to-End Machine Learning Project


- How does the company expect to use and get benefit from the model

- Problem framing needs to be driven by the objective, not other wat around

- Pipelines are processes that run models and read.dump results to data stores
    - Components often run asynchronously

- Identifying a current solution helps with benchmarking and inspiration

- Checking assumptions helps to uncover issues and inform project design

- Root Mean Squared Error (RMSE) is common measure of error of the system's predictions
    - "Take the distance from predicted to actual value and square it"
    - "Take the average of all these squared values"
    - "Take the square root of the average"
    
- Mean Absolute Error (MAE) is useful in other contexts
    - "Take the absolute value of the distance between y_hat and y"
    - "Take the average of all these absolute values"
    
- Both of these measures are generalized by the concept of a norm
    - RMSE is the L2-norm
    - MAE is the L1-norm
    - k-norm is the k-th root of the sum of taking the kth power of all elements
    - 0-norm gives the number of nonzero elements in a vector
    - inf-norm returns the maximum absolute value in the vector

- The higher the norm-index, the more large values are overweighted

- A subtle gotcha in constructing a test set for data that changes over time:
    - Can't do `np.random.seed()` b/c won't consistently pick same instances
    - Can't just `df.to_csv()` because need to allow for data changing
    - Solution: hash the unique ID; pick for test if hash in 20th percentile
    - If there's no unique ID, can create on using most stable feature(s)
    
- Random sampling to create a test set works only under certain circumstances:
    - Dataset is large enough (compared to number of features)
    - Stratified sampling used to remove surface area for sampling bias
    - Divide data into homogeneous subgroups (strata) and sample proportionally
    - To do this for a continuous variable, need to bin values to create strata
        - Large strata ensure a sufficient # of instances in the test set
        - Smaller strata enable more faithful representation of distribution
        - Thus, the need to pick a # of strata not-to-large and not-too-small
    -   