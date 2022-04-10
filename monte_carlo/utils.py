import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

action2arrows_map = {
    0: '←',
    1: '↓',
    2: '→',
    3: '↑'
}

def action2arrows(action):
    return action2arrows_map[action]

action2arrows_vect = np.vectorize(action2arrows)

def plot_valuefunc_and_policy(Q: np.ndarray, matrix_shape=(8,8), savefig=True):
    """
    Plot the value function v and the greedy policy with respect to v
    from the action-state value function.

    @TODO: implement the possibiliity to have multiple arrows in the same
    cell (when multiple actions are greedy).

    Parameters
    ----------
    Q : np.ndarray
        The action-state value function.
    matrix_shape : tuple
        The shape of the grid-world environment.
    savefig : bool
        Whether to save the figure. Default is True and output is saved to
        ./output/ folder.
    """
    v = np.max(Q, axis=0).reshape(*matrix_shape)
    arrows = action2arrows_vect(np.argmax(Q, axis=0).reshape(*matrix_shape))
    
    fig, axs = plt.subplots(1, 2, figsize=(13,5))

    ax = axs[0]
    ax.set_title('Value function')
    sns.heatmap(v, cmap='YlGnBu', annot=True, fmt='.2f', ax=ax)

    ax = axs[1]
    ax.set_title('Greedy policy')
    sns.heatmap(v, cmap='YlGnBu', annot=arrows, fmt='s', ax=ax)

    if savefig:
        plt.savefig('output/v_and_policy.png')
    #plt.show()
