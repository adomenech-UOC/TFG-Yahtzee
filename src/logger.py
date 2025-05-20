import time
import matplotlib.pyplot as plt

debug = False


def log(message):
    if (debug):
        print(message)


def print_time(start):
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))


def print_train_results(results, name):
    plt.plot(results)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(name)
    plt.show()


def print_eval_results(n_plays, avg_score, median_score):
    print(f"Evaluation results:")
    print(f"Score over {n_plays} games")
    print(f"Avg: {avg_score:.1f}, Median: {median_score:.1f}")
