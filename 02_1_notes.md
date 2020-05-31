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

- *What does (doesn't) the correlation coefficient tell you?* (Q)
    - Captures degree to which "X and Y move together above/below their means"
        - Covariance is expected value of (diff X from mu_x * diff Y from mu_y)
        - Correlation is covariance over (std_x times std_y)
    - Scale free measure (makes sense bc of division by stddev in formula)
    - Doesn't necessarily imply that X, Y have linear relationship!
    - Can miss dependent relationship that is also nonlinear ("fooled by 0")!
    - If X, Y have linear relationship, correlation says nothing about slope!
    
- The process of EDA and feature engineering is *iterative*

- *Why is important to write functions to prepare data for your algorithm?* (Q)
    - Enables the exact same transformations to be done on new data
    - Can reuse component parts in later projects
    - Can use pipeline in production to feed live data to fitted model
    - Enables easier iteration to find the best pipeline for the task
    
- *What are some ways of dealing with missing features?* (Q)
    - Drop rows with missing values; risks losing a lot of data if not careful
    - Drop the feature; makes most sense if most of the values missing
    - Impute some value, e.g. 0, 1, -1, mean, median, etc.
    - Need to make sure that _exact_ same logic is applied to test set, PROD
    
- The genius of `sklearn`'s API is the consistency across a small no. of objects
    - *Estimator*: any object that can estimate some parameters based on data
        - Cause an estimator to estimate parameters with `.fit()` method
        - Estimator's hyperparameters stored in object's attributes
        - learned parameters also available (with trailing_underscore)
    - *Transformer*: any object that can produce a new dataset from an existing
        - Called with `.trasnform()` method
    - `.fit_transform()` is logically equivalent to `fit` then `transform`
        - Implementation of `fit_transform` may be more efficient 
    - Lot of overlap of objects that are both estimators and transformers
    - Predictors are any object that makes a prediction
        - Also a lot of overlap of estimators that are also predictors
        - Predictors also implement `.score()` method measure prediction quality
        - Feed `X_test` to `.predict()` and `y_test` to `.score()`
    - Data is represented as `numpy` / `scipy` arrays and matrices
    - `Pipeline` created by composing arbitrary transformers and final estimator
        - `Pipeliine` itself is an estimator
  
- *When does it make sense to encode a categorical feature as ordinal?* (Q)      
    - Makes sense when there is an ordering of feature's values

- *When does it make sense to encode a categorical feature as dummies?* (Q)
    - Makes some sense when there is no clear ordering or distance btwn values
    
- *What are some strategies to deal with high-cardinality features?* (Q)
    - Find ways to convert features to numeric
    - Replace feature with a learnable, low-dimensional embedding