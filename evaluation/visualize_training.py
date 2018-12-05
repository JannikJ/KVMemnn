import numpy
import re


SHOW = True


def visualize(time_steps, remaining_times, iteration_steps, percentages, losses, val_accuracies):
    from matplotlib import pyplot as plt
    from itertools import cycle

    # plt.figure(1)
    # plt.clf()
    fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    yticks = numpy.arange(0, 3.5, 0.2)
    yrange = (yticks[0], yticks[-1])
    xticks = numpy.arange(0, 100000, 20000)
    xrange = (xticks[0], xticks[-1])
    ax0.set_title("Loss")
    ax0.set_xticks(xticks)
    ax0.set_yticks(yticks)
    ax0.set_ylim(yrange)
    ax0.set_xlim(xrange)
    yticks_2 = numpy.arange(0, 1.0, 0.1)
    yrange_2 = (yticks_2[0], yticks_2[-1])
    ax1.set_title("Validation Accuracy")
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks_2)
    ax1.set_ylim(yrange_2)
    ax1.set_xlim(xrange)
    generated_colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (0, 0, 0, 0.5)]
    for k in range(0, len(time_steps), 3):
        # plt.plot(coords[3628 - (404 - i)][0], coords[3628 - (404 - i)][1], col + '.')
        # plot loss
        ax0.plot(int(iteration_steps[k]), float(losses[k]), '.',
                 color=generated_colors[0][0:3],
                 alpha=1, markersize=3)
        # plot accuracy
        ax1.plot(int(iteration_steps[k]), float(val_accuracies[k]), 'o',
                 color=generated_colors[1][0:3],
                 alpha=1, markersize=3)
        # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #          markeredgecolor='k', markersize=14)
    # plt.title('Training phase of chatbot')
    plt.savefig("Chatbot training.pdf", width=1920, height=1080)
    if SHOW:
        plt.show()
    else:
        plt.clf()


if __name__ == '__main__':
    file_name = "../log-2711-schedule-train.txt"
    time_steps = []
    remaining_times = []
    iteration_steps = []
    percentages = []
    losses = []
    val_accuracies = []
    with open(file_name, 'r') as f:
        for line in f:
            if re.match("[0-9]", line[0]):
                times = re.match("([0-9]+)m ([0-9]+)s.*", line).group(1, 2)
                time_steps.append(times[0] * 60 + times[1])
                times = re.match("[0-9]+m [0-9]+s \(- ([0-9]+)m ([0-9]+)s\).*", line).group(1, 2)
                remaining_times.append(times[0] * 60 + times[1])
                iteration_steps.append(re.match("[0-9]+m [0-9]+s \(- [0-9]+m [0-9]+s\) \(([0-9]+).*", line).group(1))
                percentages.append(re.match("[0-9]+m [0-9]+s \(- [0-9]+m [0-9]+s\) \([0-9]+ ([0-9]+).*", line).group(1))
                losses.append(re.match("[0-9]+m [0-9]+s \(- [0-9]+m [0-9]+s\) \([0-9]+ [0-9]+%\) ([0-9]+\.[0-9]+).*", line).group(1))
                val_accuracies.append(re.match("[0-9]+m [0-9]+s \(- [0-9]+m [0-9]+s\) \([0-9]+ [0-9]+%\) [0-9]+\.[0-9]+"
                                               " - val_accuracy ([0-9]+\.[0-9]+).*", line).group(1))
    visualize(time_steps, remaining_times, iteration_steps, percentages, losses, val_accuracies)
    medium_loss = 0
    medium_acc = 0
    for i in range(10):
        values = 100
        for k in range(values):
            try:
                medium_acc += float(val_accuracies[i*values + k])
                medium_loss += float(losses[i*values + k])
            except:
                values = k
                print("ERROR")
                break
        try:
            medium_acc = medium_acc / values
            medium_loss = medium_loss / values
            print("ITERATION " + str(i) + ": " + str(medium_acc) + "; " + str(medium_loss))
            medium_loss = 0
            medium_acc = 0
        except ZeroDivisionError:
            print("ZERO DIVISION")
            break
