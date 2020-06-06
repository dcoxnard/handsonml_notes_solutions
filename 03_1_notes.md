# Chapter 3: Classification on MNIST

- `sklearn`'s `SGDClassifier` implements many estimators
    - Tied together through common thread of SGD optimization strategy
    - Many models have alternate implementations available through `sklearn` API
    - Same goes for `SGDRegressor`

- Accuracy as a model evaluation metric:
    - Describes what percentage of the time a classifier gets the correct class
    - Advantage: very easy to interpret
    - Disadvantage: misleading for skewed datasets
    - Disadvantage: Incomplete picture if FP, FN errors carry different costs
    
- Confusion matrix for model evaluation:
    - Advantage: carries more information about model performace
    - Disadvantage: can be difficult to summarize in a one-liner
    - Disadvantage: interpretation gets more difficut with the number of classes
    
- Precision: TP / (TP + FP)
    - "What % of the time is a prediction of True correct?"
    - Measures "purity" of True predicitons
    
- Recall TP / (TP + FN)
    - "What % of real True's get classified at such?"
    - Measures "dependability" of classifier to detect True cases
    - Statisticians know it as "sensitivity"
    
- Precision/Recall Tradeoof:
    - In general, there will be a tradeoff between precision and recall
    - A good model will "move the frontier" of teh tradeoff
    - This idea can be visualized with an ROC graph
    - Trading off between P and R is done by varying the decision boundary
    - Recall can only decrease as the threshold is raised (monotonic)
    - Precision may actually decrease as threshold is raised
    
- Note about specificity and sensitivity:
    - Recall and Sensitivity are the same
        - memory device: "RE" and "SE" have "E" as second letter
        - They are calculated using *only* the predicted labels, no ground-truth
    - Specificity is almost like the "mirror image" of precision
        - They are mirror images in (only) the following sense:
            - Precision applies to True predictions
            - Specificity applies to negative predictions
        - However, they are NOT mirror images in the following sense:
            - Numerator of Precision is *true* positives
            - Numerator of Specificity is *false* negatives
            - THus, both are constructed so that a higher number is "good"
            
- F1-Score is the harmonic mean of precision and recall:
    - 2 / (1/P + 1/F)
    - Harmonic mean gives more weight to low values than the arithmetic mean
        - Convex curve that's tangent to the arithmetic mean at P = R
    - Thus it is only possible to get a high F1-score when P and  R both high
    - Favors models with similar precision and recall
    - Can instead use weighted harmonic mean to cappture different tradeoff
        - F-beta score 

- Relative (and absolute) costs of FPs vs FN's depends on the problem at hand

