# Chapter 1 Exercises

1. How would you define Machine Learning?

Machine Learning is the act of training a computer to perform a task
such that it infers patterns from data and is not explicitly programmed.

2. Can you name four types of problems where it shines?

The chapter called out spam filtering and speech recognition, among others.
Some more are:
    - Identifying certain types of objects in an image
    - Navigating autonomously through an environment (e.g. driving, flying)
    - Tagging certain parts of speech in a text
    - Detecting fraudulent patterns of use (e.g. credit card, product reviews)
    
3. What is a labeled training set?

A dataset is labeled if each instance (row) is associated with a label, or
a value to be learned and later estimated.  A labeled training set is used
by a Machine Learning system to learn a desired rule for predicting labels.

4. What are the two most common superviseed tasks?

The two most common supervised tasks are regression, which involves
predicting a continuous numerical label, and classification, which entails
predicting a categorical label.  Both tasks are supervises because they involve
making predictions of labels.

5. Can you name four common unsupervised tasks:

Some unsupervised tasks are:
    - Clustering
    - Dimensionality reduction
    - Association rule learning
    - Anomaly detection
    - Topic modeling (not mentioned in the chapter)
    
6. What type of Machine Learning algorithm woud you used to segment your
customers into multiple groups?

Clustering is the correct approach for segmenting customers because it involves
discovering patterns within the customer base that describe segments where
customers are similar to each other while being systematically different from
customers in other segments.

8. Would you frame the problem of spam detection as a supervised learning
problem or an unsupervised learning problem?

Spam detection is a supervised learning problem.  The problem is characterized 
by an assignment of spam/not-spam to every email.  The goal of spam detection
is to predict which label a new email should be associated to.  Since this
involves predicting a ground-truth value, rather than uncovering a pattern
without access to ground-truth labels, spam detection is a supervised learning
problem.

9. What is an online learning system?

An online learning system learns incrementally from each new piece of data,
while being deployed in production.  The speed at which the system may learn
is governed by a learning rate.  Online learning systems contrast with
other systems that need to be taken "offline" to train on new data and 
subsequently redeployed.  A limitation of online learning is its susceptibility
to incoming bad data, whether such data is due to an accident, an attack, or
random chance.

10. What is out-of-core learning?

Out-of-core learning is a paradigm for training a machine learning system in
which the training data is too large to fit into memory.  The approach is
characterized by its solution to this problem, namely batch training which
only uses a small enough portion of the training data at once that can fit into
memory.

11. What type of learning algorithm relies on a similarity measure to make
predictions?

Instance-based learning systems use a similarity measure to make predictions
by comparing a new piece of data to all pieces of training data.  This approach
contrasts with model-based learning, which seeks to use the training data to
learn a model, i.e. a simplified, parameterized version of the world, that is
then applied to new data in order to make predictions.

12. What is the difference between a model parameter and a learning algorithm's
hyperparameter?

a model parameter is a piece of information that is tuned during the training
process by means of "fitting the model', that is, optimizing predictions against
a numerical goal.  A hyperparameter, on the other hand, is part of the model
specification and thus must be set before the model and its parameters are fit.
Unlike model parameters, which may minimize loss analytically or numerically,
the most optimal value of a hyperparameter but be selected by fitting many
models, each with different values for the hypterparameters, and selecting the
best one.

13. What do model-based learning algorithms search for?  What is the most common
strategy they use to succeed?  How do they make predictions?

A model-based learning system makes assumptions about the training data that
simplify the environment as a set of parameters.  The system then uses the
training data to find the "best" values of these parameters.  This search is
usually done by minimizing a loss function or maximizing a gain function.
To make new predictions, the learning system applies the learned model to a
new data instance.

14. Can you name four of the main challenges in Machine Learning?

Some challenges mentioned in the chapter:
    - Overfitting: fitting the data so well that the model does not generalize
    - Underfitting: using a model architecture that is not flexible enough
    - Not enough data: increases the difficulty of identifying patterns
    - Poor quality data: outliers, errors etc. reduce a model's learning power
    
15. If your model performs great on the training data but generalizes poorly
to new instances, what is happening?  Can you name three possible solutions?

Good performance on training data but poor performance on new data is the
hallmark of overfitting.  Overfitting can be mitigated by choosing a more
constrained model, such as one with a regularization penalty or a neural network
with dropout layers.  Overfitting can also be combatted by reducing the
number of features fed to the model, whether through dimensionality reduction
techniques, feature selection techniques, methods like the LASSO, hand-picking
features to exclude, or reducing the polynomial degree of a linear model.
Finally, gathering more training data helps a model to more clearly distinguish
signal from noise, reducing the potential to overfit.

16. What is a test set, and why would you want to use it?

A test set is a dataset of as-yet-unseen data that a learning system has not 
had access to while training.  A test set is meant to mimic the data that the
learning system will be faced with in production environment, thus giving an
indication of how well the model will perform once deployed.  To this end, a
test set must be high quality data that is truly representative of what the
model will face in production and not suffer from selection or sample bias,
and the learning system must not have been allowed to train on any of the test
set ("data leakage").  The advantage of a test is that it enables model
developers to get a reliable readout on how well the model can be expected to
perform, without having to actually deploy the model and run any associated
risks.  Moreover, a test set can serve as a consistent method of measuring
different models' predictive power and enables comparisons between the two.

17. What is the purpose of a validation set?

A validation set is needed for situations where model architecture needs to be
tuned that is not directly fit as part of model training.  Examples of such
architecture choices are a model's hyperparameters or the number of layers in 
a deep neural network.  To tune these characteristics of the learning system,
the model must be fit and evaluated for many different types of architectures,
e.g. for many different values of a hyperparameter.  Since generalization is the
ultimate goal, training error alone is not an acceptable way to evaluate model
performance; rather, a validation set is used, which is not used to train the
model but rather used to estimate generalization error for a given value of a
hyperparameter.

18. What is the train-dev set, when do you need it, and how do you use it?

a train dev set is a tool for model validation.  It is useful in the situation
in which the developer has access to a lot of training data, but comparatively
little data that is truly representative of the production environment and that
can thus be used in test and validation sets.  A train-dev set is a held-out
portion of the training data that is used to evaluate the model at the same time
that the validation set is used.  Using a train-dev set enables the developer
to understand the source of poor model performance: poor performance on the
train-dev set is a signal that the model is overfitting the training data.  By
contrast, good performance on the train-dev set but poor performance on the
(limited, high-quality) validation set indicates that poor model performance is
due to the differences between the training data and the testing data, as
opposed to model overfit.  Without a train-dev set, it is difficult for a
developer to discern which of these two causes leads to poor performance on a
validation set.  

19. What can go wrong if you tune hyperparameters using the test set?

Using the test set at any stage before the final model evaluation is data
leakage, which corrupts the ability to fairly judge how well the model performs
on as-yet-unseen data.  If a model is allowed to tune hyperparameters (or, just
as bad, train directly) on the test set, then the model will fit to the test
set and the final test performance will be an inaccurate picture of the model's
generalization power.  In simple terms, it allows the model to "cheat on the
test."
