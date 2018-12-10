import numpy
import re


SHOW = True


def visualize(time_steps, iteration_steps, losses, val_accuracies, plt_names, xticks, yticks_1, yticks_2, task):
    from matplotlib import pyplot as plt
    from itertools import cycle

    # plt.figure(1)
    # plt.clf()
    fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    yrange = (yticks_1[0], yticks_1[-1])
    xrange = (xticks[0], xticks[-1])
    ax0.set_title(plt_names[0])
    ax0.set_xticks(xticks)
    ax0.set_yticks(yticks_1)
    ax0.set_ylim(yrange)
    ax0.set_xlim(xrange)
    yrange_2 = (yticks_2[0], yticks_2[-1])
    ax1.set_title(plt_names[1])
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks_2)
    ax1.set_ylim(yrange_2)
    ax1.set_xlim(xrange)
    generated_colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (0, 0, 0, 0.5)]
    if mode == "chatbot":
        step_size = 3
    else:
        step_size = 1
    for k in range(0, len(losses), step_size):
        if mode == "chatbot":
            # plot loss
            ax0.plot(int(iteration_steps[k]), float(losses[k]), '.',
                     color=generated_colors[0][0:3],
                     alpha=1, markersize=3)
            # plot accuracy
            ax1.plot(int(iteration_steps[k]), float(val_accuracies[k]), '.',
                     color=generated_colors[1][0:3],
                     alpha=1, markersize=3)
        else:
            # plot loss
            ax0.plot(k+1, float(losses[k]), '.',
                     color=generated_colors[0][0:3],
                     alpha=1, markersize=3)
            # plot accuracy
            ax1.plot(k+1, float(val_accuracies[k]), '.',
                     color=generated_colors[1][0:3],
                     alpha=1, markersize=3)
    plt.savefig(mode + " " + task + " training.pdf", width=1920, height=1080)
    if SHOW:
        plt.show()
    else:
        plt.clf()


def main(task, file_names=["../log-2711-", "../log-0412-"]):
    if task == "schedule":
        num_iterations = 261
    elif task == "navigate":
        num_iterations = 450
    else:
        num_iterations = 384
    if mode == "chatbot":
        file_name = file_names[0] + task + "-train.txt"
    else:
        file_name = file_names[1] + task + "-train-phase.txt"
    time_steps = []
    remaining_times = []
    iteration_steps = []
    percentages = []
    losses = []
    val_accuracies = []
    val_losses = []
    with open(file_name, 'r') as f:
        for line in f:
            if mode == "chatbot":
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
            else:
                if re.match(str(num_iterations) + "/" + str(num_iterations) + ".*", line):
                    time_steps.append(re.match(str(num_iterations) + "/" + str(num_iterations) + " \[=*\] - ([0-9]+)s .*", line).group(1))
                    losses.append(re.match(str(num_iterations) + "/" + str(num_iterations) + " \[=*\] - [0-9]+s [0-9]+s/step"
                                                                                       " - loss: ([0-9]+\.[0-9]+).*", line).group(1))
                    val_losses.append(re.match(str(num_iterations) + "/" + str(num_iterations) + " \[=*\] - [0-9]+s [0-9]+s/step"
                                                                                       " - loss: [0-9]+\.[0-9]+ - val_loss: ([0-9]+\.[0-9]+).*", line).group(1))
    if mode == "chatbot":
        xticks = numpy.arange(0, 140000, 20000)
        yticks = numpy.arange(0, 3.5, 0.2)
        yticks_2 = numpy.arange(0, 1.0, 0.1)
        visualize(time_steps, iteration_steps, losses, val_accuracies, ["Loss", "Validation Accuracy"], xticks, yticks, yticks_2, task)
    else:
        xticks = numpy.arange(0, 40, 10)
        yticks = numpy.arange(0, 7.0, 0.5)
        yticks_2 = numpy.arange(0, 7.0, 0.5)
        visualize(time_steps, iteration_steps, losses, val_losses, ["Loss", "Validation Loss"], xticks, yticks, yticks_2, task)
    if mode == "chatbot":
        medium_loss = 0
        medium_acc = 0
        for i in range(100):
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
                print("ITERATION " + str(i) + ": Accuracy: " + str(medium_acc) + "; Loss: " + str(medium_loss))
                medium_loss = 0
                medium_acc = 0
            except ZeroDivisionError:
                print("ZERO DIVISION")
                break
    else:
        medium_loss = 0
        medium_val_loss = 0
        for i in range(100):
            values = 10
            for k in range(values):
                try:
                    medium_val_loss += float(val_losses[i*values + k])
                    medium_loss += float(losses[i*values + k])
                except:
                    values = k
                    print("ERROR")
                    break
            try:
                medium_val_loss = medium_val_loss / values
                medium_loss = medium_loss / values
                print("ITERATION " + str(i) + ": Val loss: " + str(medium_val_loss) + "; Loss: " + str(medium_loss))
                medium_loss = 0
                medium_val_loss = 0
            except ZeroDivisionError:
                print("ZERO DIVISION")
                break


if __name__ == '__main__':
    full = True
    if full:
        for mode_local in ["gan", "chatbot"]:
            mode = mode_local
            for task_name in ["schedule", "navigate", "weather"]:
                main(task_name)
    else:
        mode = "chatbot"
        main("original", file_names= ["../final-original/log-1210-", ""])
