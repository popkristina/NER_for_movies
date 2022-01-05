import matplotlib.pyplot as plt


def plot_learning_curves(hist, curve1, curve2):
    """
    Intended to plot accuracy and loss curves alongside.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(hist[curve1])
    plt.plot(hist[curve2])
    plt.show()
    