import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_models(models: list, test_location: str, save_file=None, sns_style='darkgrid', sns_context='talk', palette='coolwarm', labels=None):
    """
    Skylar slide style plot for classification nets. Plots from saved Keras model files.

    models: List of file paths for saved models.

    test_location: Directory path for test data.

    save_file: File path to save plot to. Passing None will not save the plot.

    labels: Dictionary of names for the models, with model path as key and replacement name as value.

    Also takes additional seaborn settings.
    """
    idg = ImageDataGenerator(rescale=1./255)

    test_set = idg.flow_from_directory(test_location,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='binary',
                                            color_mode='grayscale')

    sns.set_style(sns_style)
    sns.set_context(sns_context)

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.set_tight_layout(True)

    xticklabels = [labels[name] for name in models] if labels else models
    accuracies = []
    recalls = []

    for model in models:
        loaded_model = keras.models.load_model(model)
        loss, accuracy, recall = loaded_model.evaluate(test_set)
        accuracies.append(accuracy)
        recalls.append(recall)

    sns.barplot(x=xticklabels, y=accuracies, palette=palette)
    ax.set(ylim=(0, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    ax2 = ax.twinx()
    sns.lineplot(x=xticklabels, y=recalls, linewidth=5, palette='r')
    ax2.set_ylim((0.5,1))

    ax.set_ylabel('Accuracy Score')
    ax2.set_ylabel('Recall Score')
    ax.set_title('Model Effectiveness');

    if save_file:
        plt.savefig(save_file)
