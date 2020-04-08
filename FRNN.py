import FuzzyRough as fr
import numpy as np


def frnn(df, sample, K, R, lower_approx, upper_approx, class_range=[0, 1, 2, 3]):
    """Predicts the label of the sample according to the frnn algorithm given the necessary parameters

    Parameters:
        df (pandas DataFrame):  The training samples, with columns ["Vector"] and ["Label"]
        sample (pandas Series): The sample to be predicted
        K (int>0): The number of nearest neighbours to be used
        R (function): The fuzzy tolerance relation
        lower_approx (function): The lower approximation method
        upper_approx (function): The upper approximation method
        class_range (list): The class labels

    Returns:
        (int): The predicted class of the sample

    """
    # Getting the k-nearest neighbours according to the relation R
    distances = df["Vector"].apply(lambda x: R(x, sample.at["Vector"]))
    top_k = distances.sort_values(ascending=False)[:K]
    universe = df.loc[top_k.index]

    # returning the class for which the average membership to the fuzzy rough set of that class is the highest
    return np.argmax([(lower_approx(sample, universe, i) + upper_approx(sample, universe, i))/2 for i in class_range])


def frnn_owa(df, sample, K, R, implicator, t_norm, w_low, w_upp, class_range=[0, 1, 2, 3]):
    """Generates the lower and upper approximation methods, calls the frnn algorithm with these methods

    Parameters:
        df (pandas DataFrame):  The training samples, with columns ["Vector"] and ["Label"]
        sample (pandas Series): The sample to be predicted
        K (int>0): The number of nearest neighbours to be used
        R (function): The fuzzy tolerance relation
        implicator (function) : The implicator
        t_norm (function) : The t_norm
        w_low (vector): The lower approximation weighting scheme
        w_upp (function): The upper approximation weighting scheme
        class_range (list): The class labels

    Returns:
        (int): The predicted class of the sample
    """

    def lower_approx(y, universe, present=False):
        return fr.lower_approx_owa(y,
                                   universe,
                                   implicator,
                                   R,
                                   lambda y: int(y["Label"] == present),
                                   w_low(K),
                                   )

    def upper_approx(y, universe, present=False):
        return fr.upper_approx_owa(y,
                                   universe,
                                   t_norm,
                                   R,
                                   lambda y: int(y["Label"] == present),
                                   w_upp(K),
                                   )
    return frnn(df, sample, K, R, lower_approx, upper_approx, class_range)