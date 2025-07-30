import numpy as np

import matplotlib.pyplot as plt


x_clipped = [-2, -1, 0, 1, 2]
x = [ -4, -3, -2, -1, 0, 1, 2, 3, 4]



def normalize(x):
    total = sum(x)
    for i in np.arange(0, 9, 1):
        x[i] = x[i] / total
    return x


def norm_dist(x, mu, sigma):
    prob = []
    for i in range(len(x)):
        p = 1 / (np.sqrt(2 * np.pi * sigma**2)) * np.exp( - (x[i] - mu)**2 / (2 * sigma**2))
        prob.append(p)
    return normalize(prob)


def get_sd(changeability):
    epsilon = 1e-6  # avoid division by 0
    changeability = 1 - changeability
    sig = 0.7985695902042361 * (0.5 / (changeability + epsilon)) # change = 0.7985695902042361 is roughly sig = 0.5 for 0 when normalized
    sig = np.clip(sig, 0.2, 8)
    return sig


def calc_tendency (weight, base_prob):
    #prep weight
    weight = int(np.clip(weight, -2, 2))
    weight = weight * (-1)

    # shift array by 1 or 2 spots, then perform linear interpolation
    shift = np.roll(base_prob, weight)
    blend_factor = 0.4

    new_dist = [(1-blend_factor) * original + blend_factor * shifted for original, shifted in zip(base_prob, shift)]
    return normalize(new_dist)


def clip_and_norm(dist):
    clipped = []
    for i in range(5):
        clipped.append(dist[i+2])
    total = sum(clipped)
    for i in range(5):
        clipped[i] = clipped[i] / total
    return clipped

    
def change_prob(interpreted_user_icm, active_llm_icm, strategy = "mimic", changeability=0.5):

    if strategy == "complement":
        interpreted_user_icm = int(round(1 * (4 - interpreted_user_icm) + (1 - 1) * interpreted_user_icm))
    
    llm_icm = active_llm_icm - 2
    user_icm = interpreted_user_icm -2
    sigma = get_sd(changeability)
    mu = llm_icm

    if (llm_icm - user_icm) == 0:
        y = clip_and_norm(norm_dist(x, mu, sigma))
        #plot_dist(x_clipped, y)
    else:
        y = clip_and_norm(calc_tendency(llm_icm-user_icm, norm_dist(x, mu, sigma)))
        #plot_dist(x_clipped,y)

    new_val = np.random.choice(range(5), p=y)
    return new_val, y




def plot_dist(x, y):
    #plt.bar(x, y)
    plt.plot(x,y)
    plt.show() 


if __name__ == "__main__":
    y = change_prob(4, 0, 0.5)
    plt.plot(x_clipped,y)
    plt.show() 