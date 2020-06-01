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

- Test set must be created right away to prevent data leakage and snooping
    - Per-subset test sets can help gain a deeper understanding of the data

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
        - Names have to be unique and can't contain double underscores
        `Pipeline` exposes the methods of its final estimator
  
- *When does it make sense to encode a categorical feature as ordinal?* (Q)      
    - Makes sense when there is an ordering of feature's values

- *When does it make sense to encode a categorical feature as dummies?* (Q)
    - Makes some sense when there is no clear ordering or distance btwn values
    
- *What are some strategies to deal with high-cardinality features?* (Q)
    - Find ways to convert features to numeric
    - Replace feature with a learnable, low-dimensional embedding
    
- Any class with.fit(), .transform(), and .fit_transform can be used in Pipeline
    - Inherit from `BaseEstimator` to get `.get_params()` and `.set_params()`
    - Inherit from `TransformerMixin` to get auto-implemented `.fit_transform()`
    - Choosing hyperparameters helps figure out which transforms are useful

- *What are common ways to scale features? What are their (dis)advantages?* (Q)
    - Normalization (min-max scaling) makes a feature range from 0 to 1
        - sklearn MinMaxScaler
    - Standardization sets mean to 0 and standard deviation to 1
        - sklearn StandardScaler
    - Standardization doesn't bound the values, which can be problem for NNs
    - Normalization is susceptible to outliers

- *How must feature scaling fit into overall data transformation process?* (Q)
    - Don't need to scale the target, just the features
    - Transformer must be fit *only* on train set, and not refit on test set    

- *Which algorithms (don't) require feature scaling, and why?* {Q)

- K-fold cross validation gives some estimates of the test error
    - Stddev of scores gives estimate of the spread of the estimate
    - Train on (k-1)/k rows of the data, and evaluate on 1/k rows
    - Repeat the process k times and collect the resulting scores
    - sklearn requires a utility function, not a cost function
    - Tradeoff of K-fold cross validation is that it takes k times as long
    
- Recommended to first try out some various models to narrow to 2-5 candidates
    - Don't spend too much time tweaking hyperparameters
    
- `joblib` library can be more efficient than pickle for large numpy arrays

- *What are the tradeoffs in using `GridSearchCV` or `RandomizedSearchCV`*? (Q)
    - Grid search offers more systematic information about the search space
    - Randomized search enables search through a high-dimensional HP-space
    - Randomized search enables a simpler specification of "computation budget"
    
- Using ensemble methods makes sense if different models make different mistakes

- 95% CI of estimate of test error helps understand model performance

- Post-modeling steps that are not to be forgotten:
    - Present solution, capabilities and limitations, what worked + what didn't
    - Document everything
    - Write tests
    - Tidy up data visualizations for presentation

- Some ways t deploy the final model:
    - Within a web app: model is contained within the application code
    - As a standalone web service behind a RESTful API
        - More modular, decouples from logic/language of web app
        - Easier / safer to maintain, upgrade,3 scale and load-balance
    - Deploy using Google Cloud AI Platform and Google Cloud Storage
        - Cookie-cutter approach to deploying standalone web service
        - Scaling and load-balancing are handled under the hood
        
- Set up monitoring to detect sharp failures and "model rot"
    - In some situations, model rot is easy to detect by looking downstream
        - E.g. % of products sold that were recommended to a user
        - Easy to automate this logic
    - Sometimes human intervention is the only way to monitor performance
        - Sample for human-review QA, especially low-confidence predictions
        - May require specialists or may be achievable with e.g. Mechanical Turk
    - Don't forget to define the process: what to do when model fails
    
- It is possible to automate a lot of the DS lifecycle, if inputs are well-known
    - Data collection
    - Model retraining / hyperparameter search
    - Model evaluation (e.g. dump current and challenger scores to a report)
    - Deployment into production if challenger model is better
    - Model versioning, archival and rollback
    - Data versioning, archival, comparison and (some) QA
