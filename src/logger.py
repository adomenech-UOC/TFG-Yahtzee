import time
import matplotlib.pyplot as plt
import numpy as np

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
    # Generate recent 50 interval average
    average_reward = []
    for idx in range(len(results)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = results[:idx+1]
        else:
            avg_list = results[idx-49:idx+1]
        average_reward.append(np.average(avg_list))

    plt.plot(results)
    plt.plot(average_reward)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(name)
    plt.show()

def process_average(results):
    # Generate recent 50 interval average
    average_reward = []
    for idx in range(len(results)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 50:
            avg_list = results[:idx+1]
        else:
            avg_list = results[idx-49:idx+1]
        average_reward.append(np.average(avg_list))   
    
    return average_reward

def print_eval_results(n_plays, avg_score, median_score):
    print(f"Evaluation results:")
    print(f"Score over {n_plays} games")
    print(f"Avg: {avg_score:.1f}, Median: {median_score:.1f}")
