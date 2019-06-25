""" Plot Tools

This script have a sets of tools that handle metrics and plots graphs, matrix and tables.
The tools here are implemented basically by Scikit, and adapted as demanded, bellow  is possible
see the references to this fonts:

- A compilation of bests 50 matplotlib visualization, to analysis data:
    https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/

- Implementations of Histograms, Density Plots, Box and Whisker Plots, Correlation Matrix Plot and Scatterplot Matrix:
    https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

- API reference to Scikit metric:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

- API reference to Scikit plot metrics:
    https://scikit-plot.readthedocs.io/en/stable/metrics.html

This file can also be imported as a module and contains the following functions:
    * plot_confusion_matrix - plot/save and image of confusion matrix
    * get_and_plot_metrics - return the metrics of a set of data and show/save/print a table with metrics of a set of data
    * plot_distribution_data - plot a set of data in a 2D plan, showing the distribution,
      where axis x and y are 2 diff features
    * test_this_module - Function that runs a MNIST problem to test this module
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean

from sklearn.decomposition import PCA
from sklearn.utils.multiclass import unique_labels
import sklearn.metrics as skmetrics
import scikitplot as skplt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels=None, true_labels=None, pred_labels=None, title=None, normalize=False,
                          hide_zeros=False, hide_counts=False, x_tick_rotation=0, ax=None, figsize=None, cmap='Blues',
                          title_fontsize='large', text_fontsize='medium', save_image=False, plot=True, image_path="",
                          image_name=""):
    """Generates confusion matrix plot from predictions and true labels, and show and/or save in disc.

    :param y_true: (array-like, shape (n_samples)) – Ground truth (correct) target values.
    :param y_pred: (array-like, shape (n_samples)) – Estimated targets as returned by a classifier.
    :param labels: (array-like, shape (n_classes), optional) – List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
    :param true_labels: (array-like, optional) – The true labels to display. If none is given, then all of the labels are used.
    :param pred_labels: (array-like, optional) – The predicted labels to display. If none is given, then all of the labels are used.
    :param title: (string, optional) – Title of the generated plot. Defaults to “Confusion Matrix” if normalize is True. Else, defaults to “Normalized Confusion Matrix.
    :param normalize: (bool, optional) – If True, normalizes the confusion matrix before plotting. Defaults to False.
    :param hide_zeros: (bool, optional) – If True, does not plot cells containing a value of zero. Defaults to False.
    :param hide_counts: (bool, optional) – If True, doe not overlay counts. Defaults to False.
    :param x_tick_rotation: (int, optional) – Rotates x-axis tick labels by the specified angle. This is useful in cases where there are numerous categories and the labels overlap each other.
    :param ax: (matplotlib.axes.Axes, optional) – The axes upon which to plot the curve. If None, the plot is drawn on a new set of axes.
    :param figsize: (2-tuple, optional) – Tuple denoting figure size of the plot e.g. (6, 6). Defaults to None.
    :param cmap: (string or matplotlib.colors.Colormap instance, optional) – Colormap used for plotting the projection. View Matplotlib Colormap documentation for available options. https://matplotlib.org/users/colormaps.html
    :param title_fontsize: (string or int, optional) – Matplotlib-style fontsizes. Use e.g. “small”, “medium”, “large” or integer-values. Defaults to “large”.
    :param text_fontsize: (string or int, optional) – Matplotlib-style fontsizes. Use e.g. “small”, “medium”, “large” or integer-values. Defaults to “medium”.
    :param save_image: (Boolean, optional) – If True, save image in disc
    :param plot: (Boolean, optional) – If True, shows the plot
    :param image_path: (string, optional) – Path to save image, e.g.: '../results'
    :param image_name: (string, optional) – Image name
    """

    # Plot confusion matrix
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, labels, true_labels, pred_labels, title, normalize, hide_zeros,
                                hide_counts, x_tick_rotation, ax, figsize, cmap, title_fontsize, text_fontsize)

    # Save and/or show plotted matrix
    if save_image:
        plt.savefig(image_path + '/' + image_name)
    if plot:
        plt.show()


def get_and_plot_metrics(y_true, y_pred, labels=None, labels_name=None, pos_label=1, average='macro',
                         sample_weight=None, normalize=True, accuracy_score=True, f1_score=True, precision_score=True,
                         recall_score=True, plot_table=False, save_table=False, table_format='latex', table_name="",
                         file_path="", file_name=""):

    """Get metrics from predictors and true labels, and show/save table in file as latex or plain-text

    :param y_true: (array-like, shape (n_samples)) – Ground truth (correct) target values.
    :param y_pred: (array-like, shape (n_samples)) – Estimated targets as returned by a classifier.
    :param labels: (array-like, shape (n_classes), optional) – List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
    :param labels_name: (array-like[string], shape (n_classes), optional) – List of labels name to show in plots
    :param pos_label: (str or int, 1 by default, optional) – The class to report if average='binary' and the data is binary. If the data are multiclass or multilabel, this will be ignored; setting labels=[pos_label] and average != 'binary' will report scores for that label only.
    :param average: (string, [None, ‘binary’, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’], optional) –
    :param sample_weight: (array-like of shape = [n_samples], optional) – This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data.
    :param normalize: (bool (default=True), optional) – If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
    :param accuracy_score: (bool (default=True), optional) – Calculate and return accuracy score
    :param f1_score: (bool (default=True), optional) – Calculate and return f1 score
    :param precision_score: (bool (default=True), optional) – Calculate and return precision score
    :param recall_score: (bool (default=True), optional) – Calculate and return recall score
    :param plot_table: (bool (default=True), optional) – Plot and show table
    :param save_table: (bool (default=False), optional) – Save table in file
    :param table_format: (string, ['latex', 'plain-text'], optional) – Convert table to file format
    :param table_name: (string, optional) – Name of table in plots and file caption if latex format
    :param file_path: (string, optional) – Path to save file, e.g.: '../results'
    :param file_name: (string, optional) – Image name, e.g.: 'test_table_file'
    :return: (array-like, shape (n_selected_metrics)) – An array with return of metrics selected: '([accuracy_score,][f1_score,][precision_score,][recall_score])'.
    The elements inside array-like follow the return of selected metric, ex: f1_score with Average None, return an array with the scores for each class.
    See this page for more about metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """

    # return of function
    ret = []

    # Get accuracy rounded for 2 decimals
    accuracy = np.round(skmetrics.accuracy_score(y_true, y_pred, normalize, sample_weight), 2)

    # Define the metrics to be calculated, and possibles averages, and define columns name
    metrics = ('f1_score', 'precision_score', 'recall_score')
    averages = (None, 'micro', 'macro', 'weighted')
    columns = ('F1 Score', 'Precision', 'Recall')

    # Initialize the dictionary containing the metrics
    data_scores = {metric: {} for metric in metrics}

    # Calculate every metric for every average rounded for 2 decimals and save in 'data_scores'
    for metric in metrics:
        for metric_average in averages:
            data_scores[metric][metric_average] = np.round(
                getattr(skmetrics, metric)(y_true, y_pred, labels, pos_label, metric_average, sample_weight)
                , 2)

    # Create table, converting all metrics excluding the accuracy to DataFrame and add labels_name
    metrics_table = pd.DataFrame({metric: data_scores[metric][None] for metric in metrics},
                                 index=np.unique(list(y_true) + list(y_pred)) if labels_name is None else labels_name)

    # Add an empty row to separate averages and accuracy
    metrics_table.loc[''] = ['' for _ in range(len(columns))]

    # Add an row with accuracy
    metrics_table.loc['accuracy'] = [accuracy] + ['-' for _ in range(len(columns)-1)]

    # Add an row for which average
    for index in range(len(averages) - 1):
        metrics_table.loc[averages[index + 1] + ' avg'] = [data_scores[metric][averages[index+1]] for metric in metrics]

    # With all metrics calculated, append the selected metrics and average to return
    if accuracy_score:
        ret.append(np.asarray(accuracy))
    if f1_score:
        ret.append(np.asarray(data_scores['f1_score'][average]))
    if precision_score:
        ret.append(np.asarray(data_scores['precision_score'][average]))
    if recall_score:
        ret.append(np.asarray(data_scores['recall_score'][average]))

    # If plot table was selected, convert to Matplotlit Table and plot table
    if plot_table:
        cell = []

        for row in range(len(metrics_table)):
            cell.append(metrics_table.iloc[row])

        plt.table(cellText=cell,
                  rowLabels=metrics_table.index,
                  colLabels=metrics_table.columns,
                  colWidths=[0.6/len(metrics_table.columns) for _ in columns],
                  loc='center')
        plt.axis('off')
        plt.show()

    # If save table was selected, save file as format demand, converting to latex if necessary
    if save_table:
        if table_format == 'latex':
            f = open(file_path + '/' + file_name + '.tex', "w")
            f.write(metrics_table.to_latex())
            f.close()
        elif table_format == 'plain-text':
            f = open(file_path + '/' + file_name + '.txt', 'w')
            f.write(metrics_table.to_string())
            f.close()
        else:
            raise Exception("Table format argument invalid!")

    return ret


def plot_distribution_data(X, y, features_name=None, save_image=False, plot=True, image_path="", image_name=""):
    """Plot distribution/histogram of data by pair of feature

    :param X: (array-like, shape (n_samples, n_features)) – Feature set to project, where n_samples is the number of samples and n_features is the number of features.
    :param y: (array-like, shape (n_samples) or (n_samples, n_features)) – Target relative to X for labeling.
    :param features_name: (array-like[string], shape(n_features)) – Name of each feature
    :param save_image: (Boolean, optional) – If True, save image in disc
    :param plot: (Boolean, optional) – If True, shows the plot
    :param image_path: (string, optional) – Path to save image, e.g.: '../results'
    :param image_name: (string, optional) – Image name
    """
    d = pd.DataFrame(data=X, columns=features_name)

    # Add labels to data
    d['features'] = np.array(['-> '+str(label) for label in y])

    # Drop duplicates and single samples
    d = d.drop_duplicates()
    d = d[d.groupby('features')['features'].transform('count').ge(3)]

    # Plot data distribution
    try:
        sns.pairplot(d, hue='features')
    except:
        plt.clf()
        sns.pairplot(d, hue='features', diag_kind='hist')

    # Save and/or show plotted distribution
    if save_image:
        plt.savefig(image_path + '/' + image_name)
    if plot:
        plt.show()


def plot_correlation_matrix(X, features_name=None, save_image=False, plot=True, image_path="", image_name=""):
    """Plot correlation matrix of features

    :param X: (array-like, shape (n_samples, n_features)) – Feature set to project, where n_samples is the number of samples and n_features is the number of features.
    :param features_name: (array-like[string], shape(n_features)) – Name of each feature
    :param save_image: (Boolean, optional) – If True, save image in disc
    :param plot: (Boolean, optional) – If True, shows the plot
    :param image_path: (string, optional) – Path to save image, e.g.: '../results'
    :param image_name: (string, optional) – Image name
    """
    d = pd.DataFrame(data=X, columns=features_name)

    # Get correlation
    correlations = d.corr()

    # Plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, X.shape[1], 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(d.columns)
    ax.set_yticklabels(d.columns)

    # Save and/or show plotted matrix
    if save_image:
        plt.savefig(image_path + '/' + image_name)
    if plot:
        plt.show()



def plot_pca_2d(X, y, title='PCA 2-D Projection', biplot=False, feature_labels=None, ax=None, figsize=None,
                cmap='Spectral', title_fontsize='large', text_fontsize='medium', n_components=None, copy=True,
                whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None,
                save_image=False, plot=True, image_path="", image_name=""):

    """Plots the 2-dimensional projection of PCA on a given dataset.

    :param X: (array-like, shape (n_samples, n_features)) – Feature set to project, where n_samples is the number of samples and n_features is the number of features.
    :param y: (array-like, shape (n_samples) or (n_samples, n_features)) – Target relative to X for labeling.
    :param title: (string, optional) – Title of the generated plot. Defaults to “PCA 2-D Projection”
    :param biplot: (bool, optional) – If True, the function will generate and plot biplots. If false, the biplots are not generated.
    :param feature_labels: (array-like, shape (n_classes), optional) – List of labels that represent each feature of X. Its index position must also be relative to the features. If None is given, then labels will be automatically generated for each feature. e.g. “variable1”, “variable2”, “variable3” …
    :param ax: (matplotlib.axes.Axes, optional) – The axes upon which to plot the curve. If None, the plot is drawn on a new set of axes.
    :param figsize: (2-tuple, optional) – Tuple denoting figure size of the plot e.g. (6, 6). Defaults to None.
    :param cmap: (string or matplotlib.colors.Colormap instance, optional) – Colormap used for plotting the projection. View Matplotlib Colormap documentation for available options. https://matplotlib.org/users/colormaps.html
    :param title_fontsize: (string or int, optional) – Matplotlib-style fontsizes. Use e.g. “small”, “medium”, “large” or integer-values. Defaults to “large”.
    :param text_fontsize: (string or int, optional) – Matplotlib-style fontsizes. Use e.g. “small”, “medium”, “large” or integer-values. Defaults to “medium”.
    :param n_components: (int, float, None or string) – Number of components to keep. if n_components is not set all components are kept: n_components == min(n_samples, n_features)
    :param copy: (bool (default True)) – If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead.
    :param whiten: (bool, optional (default False)) – When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
    Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.
    :param svd_solver: (string {‘auto’, ‘full’, ‘arpack’, ‘randomized’}) – More about in https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    :param tol: (float >= 0, optional (default .0)) – Tolerance for singular values computed by svd_solver == ‘arpack’.
    :param iterated_power: (int >= 0, or ‘auto’, (default ‘auto’)) – Number of iterations for the power method computed by svd_solver == ‘randomized’.
    :param random_state: (int, RandomState instance or None, optional (default None)) – If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when svd_solver == ‘arpack’ or ‘randomized’.
    :param save_image: (Boolean, optional) – If True, save image in disc
    :param plot: (Boolean, optional) – If True, shows the plot
    :param image_path: (string, optional) – Path to save image, e.g.: '../results'
    :param image_name: (string, optional) – Image name
    """

    # Project data in a 2D plan
    pca = PCA(n_components, copy, whiten, svd_solver, tol, iterated_power, random_state)
    pca.fit(X)

    # Plot projected data
    skplt.decomposition.plot_pca_2d_projection(pca, X, y, title, biplot, feature_labels, ax, figsize, cmap,
                                               title_fontsize, text_fontsize)

    plt.axis('off')

    # Save and/or show plotted pca
    if save_image:
        plt.savefig(image_path + '/' + image_name)
    if plot:
        plt.show()


def test_this_module():
    """ Tests for this Module:
    Function to test all the functions tools above,
    this will train an algorithm to MNIST handwritten digits dataset,
    and use all functions defined here, to show data and metrics
    """
    from sklearn import datasets, svm, metrics

    # Get dataset
    digits = datasets.load_digits()

    # Get number of samples and the data
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create de svm classifier
    classifier = svm.SVC(gamma=0.001)

    # Train classifier with 50% of samples
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Get predictions to the rest 50% of samples
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    # This section test functions in plot tools:

    # Plot and Save matrix confusion
    plot_confusion_matrix(expected, predicted, labels=digits.target_names, title='MNIST handwritten',
                          normalize=True, plot=True, save_image=True, image_path="../results",
                          image_name="TEST_MNIST_handwritten_confusion_matrix_plot")

    # Gets, Plot metrics and Save in format 'latex' and 'plain text'
    metrics_ret = get_and_plot_metrics(expected, predicted,
                                       labels=digits.target_names,
                                       labels_name=['zero', 'um', 'dois', 'três',
                                                    'quatro', 'cinco', 'seis',
                                                    'sete', 'oito', 'nove'],
                                       plot_table=True,
                                       save_table=True,
                                       file_path="../results",
                                       file_name="TEST_metrics_table_latex",
                                       table_format='latex')

    metrics_ret = get_and_plot_metrics(expected, predicted,
                                       plot_table=False,
                                       save_table=True,
                                       file_path="../results",
                                       file_name="TEST_metrics_table_latex",
                                       table_format='plain-text')

    # Plot the distribution of data in a projection in 2d plan
    plot_pca_2d(data, digits.target,
                plot=True,
                save_image=True,
                image_path='../results',
                image_name='TEST_PCA_2D_plot')

    # Plot the distribution of data
    plot_distribution_data(X=data[:, 1:5],
                           y=digits.target,
                           plot=True,
                           save_image=True,
                           image_path='../results',
                           image_name='TEST_distribution',
                           features_name=['feature 2', 'feature 3', 'feature 4', 'feature 5'])

    # Plot correlation matrix
    plot_correlation_matrix(X=data[:, :30],
                            plot=True,
                            save_image=True,
                            image_path='../results',
                            image_name='TEST_matrix_correlation',
                            features_name=[str(x) for x in range(30)])
