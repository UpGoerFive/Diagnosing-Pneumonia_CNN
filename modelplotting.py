import matplotlib.pyplot as plt

def plot_models(models: list, plot_type='line', save_file=None):
    """
    Skylar slide style plot for classification nets. Plots from saved Keras model files.

    models: List of file paths for saved models.

    plot_type: String matching either 'line' or 'bar'. Line needs validation scores and epoch numbers for each model.
    Bar plots test scores for each model.

    save_file: File path to save plot to. Passing None will not save the plot.
    """
    plot_func = plt.line if plot_type.lower() == 'line' else plt.bar

    fig, ax = plt.subplot()

    for model in models:
        ax.plot_func()