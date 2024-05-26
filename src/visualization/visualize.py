import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores, title, xlabel, ylabel, filename):
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)

def main():
    # Example scores (replace with actual data)
    training_scores = [100, 150, 200, 180, 250, 300, 280, 320]
    evaluation_scores = [50, 70, 80, 90, 100]
    inference_scores = [20, 25, 30, 35, 40, 45]

    # Plotting training scores
    plot_scores(training_scores, 'Training Scores', 'Episode #', 'Score', 'training_scores.png')

    # Plotting evaluation scores
    plot_scores(evaluation_scores, 'Evaluation Scores', 'Episode #', 'Score', 'evaluation_scores.png')

    # Plotting inference scores
    plot_scores(inference_scores, 'Inference Scores', 'Episode #', 'Score', 'inference_scores.png')

if __name__ == "__main__":
    main()
