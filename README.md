# Heart disease detection through logistic regression 1

## The purposes of this project:
* Implement a simple logistic regression model from scratch to detect heart disease.
* Visualize the development of the cross-entropy as a function of iterations.
* Gain intuition of model performance through plotting learning curves.
* Understanding the limitations of basic logistic regression.

Dataset: [Heart disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)


The trained model is a linear model with the same dimensionality as the feature space (13).


## Run program
1. Call heart_disease(data_file) in heart_disease.py.
2. Supply arguments: heart_disease(data_file, iterations=1000, learning_rate=0.01, plot_training_costs=False, plot_learning_curves=False). Only data_file is required.



![](images/cross_entropy_plot.png?raw=true)

The above plot demonstrates the convergence of the cost for this model. The convergence value is approx. 0.33.



![](images/learning_curves.png?raw=true)

From the above plot, it seems like both the training cost and the test cost converge to the same value with increased number of training examples. From the looks of it, the model has low variance and extrapolates well to new examples.

The training and test accuracies are approx. the same (80% - 85%). This isn't nearly good enough for the problem of detecting heart disease. From the learning curves, you could hypothesize that the bias is too big. Another model is likely required to improve performance.

Precision and recall (F1 score) are normally better validators for this kind of problem. However, since the accuracy of the model is so low, there is really no point in computing these values.  



