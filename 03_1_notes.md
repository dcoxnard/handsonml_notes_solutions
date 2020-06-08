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
    
- Precision/Recall Tradeoff:
    - In general, there will be a tradeoff between precision and recall
    - A good model will "move the frontier" of teh tradeoff
    - This idea can be visualized with an ROC graph, or plotting P vs R
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
    - Can instead use weighted harmonic mean to capture different tradeoff
        - F-beta score 

- Relative (and absolute) costs of FPs vs FN's depends on the problem at hand

- ROC Curve plots true positive rate vs false positive rate
    - True positive rate is synonymous with recall and sensitivity
    - False positive rate is 1 - True negative rate
        - True negative rate is synonymous with specificity
        - Thus, ROC plots sensitivity vs 1 - specificity

- AUC summarizes "how far the tradeoff bound is pushed"
    - Area under the ROC curve
    - A perfectly random classifier (diagonal line) will have AUC = 0.5
    - A perfect classifier will have an AUC of 1
    
- *What are the (dis)advantages of using ROC instead of PR curve?* (Q)
    - Advantage: ROC curve is easier to interpret
    - Advantage: Using ROC enables easier "one-number" comparison (AUC)
    - Disadvantage: ROC obscures classifier quality when True class is rare

- Thus, PR curve is better for situations when either:
    - The True class is rare
    - False positives are more important / costly than false negatives
    
- `sklearn` classifiers have `.decision_function()` or `predict_proba()` or both

- Multiclass Classification seeks to classify 2-or-more classes
    - Supported natively by eg Random Forest, SGD Classifier, Naive Bayes
    - Not supported natively by eg Logistic Regression, SVM
    
- OvR and OvO schemes can be used to repurpose binary classifiers for MC setting
    - One-vs-Rest:
        - Train K binary classifiers
        - Create predictions with all of them
        - Choose the prediction with the highest score
    - One-vs-One:
        - Train K(K - 1) / 2 classifiers
        - Predict with all of them
        - Take "the class that wins the most duels"
            - Eg MNIST:
                - Get # of 0-vs-* estimators that predict 0
                - Get # of 1-vs-* estimators that predict 1
                - ...
                - Take class labels with higest #
        - Tradeoff: More classifiers needed but can use smaller training sets
            - Better for algorithms that scale poorly with data size, e.g. SVM
    - `sklearn` makes available `OneVsOneClassifer` and `OneVsRestClassifer`

- Multilabel classification is setting where multiple labels should be outputted
    - Not all classifiers support multilabel classification
    
- More labels to predic means more dimensions along which to evaluate models
    - One idea: Take average of metric (e.g. F1, precision) across all labels
    - Other approaches are necessary if labels are not equally important
    - If labels imbalanced can weight by support (lower weight for rarer labels)
    
- Multioutput classification is the most general setting
    - Multuple labels, and each label can take more than 2 values
    
- 
