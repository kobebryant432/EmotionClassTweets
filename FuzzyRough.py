import numpy as np
from numpy.linalg import norm
from itertools import product
# implicators
# -------------

I_kleene_dienes = lambda x, y: max(1-x, y)
I_lukasiewicz = lambda x, y: min(1, 1-x+y)

# t-norms
# --------

T_min = lambda x, y: min(x, y)
T_lukasiewicz = lambda x, y: max(0, x+y-1)

# OWA weights
# ------------

W_strict_L = lambda n: (n - 1) * [0] + [1]
W_strict_U = lambda n: [1] + (n-1)*[0]

W_av = lambda n: n*[1/n]

W_add_L = lambda n: [2*i/(n*(n+1)) for i in range(1, n+1)]
W_add_U = lambda n: [2*(n+1-i)/(n*(n+1)) for i in range(1, n+1)]

W_exp_L = lambda n: [2**(i-1)/(2**n -1) for i in range(1, n+1)]
W_exp_U = lambda n: [2**(n-i)/(2**n -1) for i in range(1, n+1)]

W_invadd_L = lambda n: [1/(n-i)*sum(1/j for j in range(1, n+1)) for i in range(n)]
W_invadd_U = lambda n: [1/i*sum(1/j for j in range(1, n+1)) for i in range(1, n+1)]



# upper and lower approximations
# -------------------------------
def lower_approx_owa(x, top_k, I, rel, A, w):
    """generates the membership of x to the lower owa approximation

    Parameters:
        x (vectors)
        top_k (lis:vectors): the k-nearest neighbours of x
        T (functions):  The t_norm
        rel (functions): The relation function
        A (functions): A membership function
        w (vector) : The owa weights

    Returns:
        (float): The resulting membership of x

    """
    arguments = [I(rel(x["Vector"], y["Vector"]), A(y)) for _, y in top_k.iterrows()]
    arguments.sort(reverse=True)
    return sum(weight*argument for weight, argument in zip(w, arguments))

def upper_approx_owa(x, top_k, T, rel, A, w):
    """generates the membership of x to the upper owa approximation

    Parameters:
        x (vectors)
        top_k (lis:vectors): the k-nearest neighbours of x
        T (functions):  The t_norm
        rel (functions): The relation function
        A (functions): A membership function
        w (vector) : The owa weights

    Returns:
        (float): The resulting membership of x

    """
    arguments = [T(rel(x["Vector"], y["Vector"]), A(y)) for _, y in top_k.iterrows()]
    arguments.sort(reverse=True)
    return sum(weight*argument for weight, argument in zip(w, arguments))

# distance functions
# -------------------

# distance between average of vectors
def distance1(v, w, order = 2):
    """Calculates the distance between vectors v and w"""
    dist = norm(v-w, order)
    return dist

# tolerance relations
# --------------------

def relation1(v, w):
    """Calculates the cossine_similarity between two vectors v and w"""
    cos_sim = v/norm(v) @ w/norm(w)
    return (cos_sim + 1)/2


# Generating distance based relations
def relation_from_d_measure(tweet_vectors, d):
    """Generating the distance based relation from the distnance measure d using the
            Parameters:
                tweet_vectors(list: vectors): The universe of vectors
                d(function): Distance measure
            Returns:
                (float)
        """
    max_dist = max(d(x, y, np.inf) for x, y in product(tweet_vectors, tweet_vectors))
    return lambda x, y: 1 - d(x, y, np.inf) / max_dist

