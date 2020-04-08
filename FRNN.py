import FuzzyRough as fr
import numpy as np


def frnn(df, sample, K, R, lower_approx, upper_approx, class_range=[0, 1, 2, 3]):
    distances = df["Vector"].apply(lambda x: R(x, sample.at["Vector"]))
    top_k = distances.sort_values(ascending=False)[:K]
    universe = df.loc[top_k.index]
    return np.argmax([(lower_approx(sample, universe, i) + upper_approx(sample, universe, i))/2 for i in class_range])


def frnn_owa(df, sample, K, R, implicator, t_norm, w_low, w_upp, class_range=[0, 1, 2, 3]):
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

def FRNN_classifier(unclassified_tweet, training_tweets, K,
                    lower_approx, upper_approx, relation):

    # get K nearest neighbours
    nearest = sorted(training_tweets,
                     key=lambda x: -relation(x, unclassified_tweet))
    K_nearest = nearest[:K]

    # class: emotion not present
    lower = lower_approx(unclassified_tweet, K_nearest, present=False)
    upper = upper_approx(unclassified_tweet, K_nearest, present=False)
    tau_not_present = (lower + upper)/2

    # class: emotion present
    lower = lower_approx(unclassified_tweet, K_nearest, present=True)
    upper = upper_approx(unclassified_tweet, K_nearest, present=True)
    tau_present = (lower + upper)/2

    return tau_present >= tau_not_present


def FRNN_FRS(unclassified_tweet, training_tweets, K,
             implicator, t_norm, relation):

    def lower_approx(y, universe, present=False):
        return fr.lower_approx_FRS(y,
                                   universe,
                                   implicator,
                                   relation,
                                   lambda y: int(y.present == present),
                                   )

    def upper_approx(y, universe, present=False):
        return fr.upper_approx_FRS(y,
                                   universe,
                                   t_norm,
                                   relation,
                                   lambda y: int(y.present == present),
                                   )

    return FRNN_classifier(unclassified_tweet,
                           training_tweets,
                           K,
                           lower_approx,
                           upper_approx,
                           relation
                           )


def FRNN_VQRS(unclassified_tweet, training_tweets, K,
              Q_most, Q_some, relation):

    def lower_approx(y, universe, present=False):
        return fr.approx_VQRS(y,
                              universe,
                              Q_most,
                              relation,
                              lambda y: int(y.present == present),
                              )

    def upper_approx(y, universe, present=False):
        return fr.approx_VQRS(y,
                              universe,
                              Q_some,
                              relation,
                              lambda y: int(y.present == present),
                              )

    return FRNN_classifier(unclassified_tweet,
                           training_tweets,
                           K,
                           lower_approx,
                           upper_approx,
                           relation
                           )


def FRNN_OWA(unclassified_tweet, training_tweets, K,
             implicator, t_norm, W_low, W_upp, relation):

    def lower_approx(y, universe, present=False):
        return fr.lower_approx_OWA(y,
                                   universe,
                                   implicator,
                                   relation,
                                   lambda y: int(y.present == present),
                                   W_low(K),
                                   )

    def upper_approx(y, universe, present=False):
        return fr.upper_approx_OWA(y,
                                   universe,
                                   t_norm,
                                   relation,
                                   lambda y: int(y.present == present),
                                   W_upp(K),
                                   )

    return FRNN_classifier(unclassified_tweet,
                           training_tweets,
                           K,
                           lower_approx,
                           upper_approx,
                           relation
                           )