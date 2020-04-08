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


# fuzzy quantifiers
# ------------------

Q_exists = lambda x: 0 if x == 0 else 1
Q_forall = lambda x: 1 if x == 1 else 0
Q_jump = lambda x, jump: 0 if x <= jump else 1


def q_linear(x, start, stop):
    if x <= start:
        return 0
    elif start < x < stop:
        return (x - start)/(stop - start)
    else:
        return 1


def q_quadratic(x, start, stop):
    if x <= start:
        return 0
    elif start < x <= (start + stop)/2:
        return 2*(x - start)**2/(stop - start)**2
    elif (start + stop)/2 < x <= stop:
        return 1 - 2*(x - stop)**2/(stop - start)**2
    else:
        return 1


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
    arguments = [I(rel(x["Vector"], y["Vector"]), A(y)) for _, y in top_k.iterrows()]
    arguments.sort(reverse=True)
    return sum(weight*argument for weight, argument in zip(w, arguments))

def upper_approx_owa(x, top_k, T, rel, A, w):
    arguments = [T(rel(x["Vector"], y["Vector"]), A(y)) for _, y in top_k.iterrows()]
    arguments.sort(reverse=True)
    return sum(weight*argument for weight, argument in zip(w, arguments))

# distance functions
# -------------------

# distance between average of vectors
def distance1(v, w, order = 2):
    dist = norm(v-w, order)
    return dist

# tolerance relations
# --------------------

def relation1(v, w):
    cos_sim = v/norm(v) @ w/norm(w)
    return (cos_sim + 1)/2


# Generating distance based relations
def relation_from_d_measure(tweet_vectors, d):
    max_dist = max(d(x, y, np.inf) for x, y in product(tweet_vectors, tweet_vectors))
    return lambda x, y: 1 - d(x, y, np.inf) / max_dist

