import matplotlib.pyplot as plt
import numpy as np
import Const as c
import json

learner_data = None
teacher_data = None


def draw_plot(ax, data, offset, edge_color, fill_color, bool):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=bool)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def main():
    global ax
    learner_data = None
    teacher_data = None
    total_iterations = 0
    with open(c.SAVE_FILE + '_teacher', 'r') as f:
        for i, each in enumerate(f.readlines()):
            lst = each.strip('[]\n').split(', ')
            if teacher_data == None:
                teacher_data = [[] for _ in range(len(lst))]
                total_iterations = len(lst)
            for ite, score in enumerate(lst):
                teacher_data[ite].append(float(score))
    with open(c.SAVE_FILE + '_learner', 'r') as f:
        for i, each in enumerate(f.readlines()):
            lst = each.strip('[]\n').split(', ')
            if learner_data == None:
                learner_data = [[] for _ in range(len(lst))]
                total_iterations = len(lst)
            for ite, score in enumerate(lst):
                learner_data[ite].append(float(score))
    fig1, ax1 = plt.subplots()
    sub_learner_data = np.array(learner_data)
    print(sub_learner_data.shape)
    draw_plot(ax1, np.array(sub_learner_data[::2500]).T, 0, 'black', 'white', False)
    plt.xticks(range(0, 96, 16), [0, 20000, 40000, 60000, 80000, 100000])
    plt.savefig('SavedData/learner_boxplot.png', bbox_inches='tight')
    fig2, ax2 = plt.subplots()
    sub_teacher_data = np.array(teacher_data)
    print(sub_teacher_data.shape)
    draw_plot(ax2, np.array(sub_teacher_data[::2500]).T, 0, 'blue', 'white', False)
    # ax.set_xticks(np.linspace(0, 100000, 10))
    # print(np.array(sub_teacher_data[::10000]).shape)
    # draw_plot(np.array(sub_teacher_data[::10]).T, +.2, 'blue', 'cyan')

    plt.xticks(range(0, 96, 16), [0, 20000, 40000, 60000, 80000, 100000])
    plt.savefig('SavedData/teach_boxplot.png', bbox_inches='tight')
    plt.show()
    length = len(learner_data[::c.EVALUATION_TIME])
    plt.plot(np.linspace(0, 100000, length), [np.mean(
        x) for x in learner_data[::c.EVALUATION_TIME]], color = 'red', label= 'Learner Fitness')
    subTD = [np.mean(
        x) for x in teacher_data[::c.EVALUATION_TIME]]
    # print(subTD)
    n = np.mean(subTD)
    subTD[0] = n
    plt.plot(np.linspace(0, 100000, length), subTD, color = 'blue', label = 'Teacher Fitness')
    plt.legend(loc = 'lower right')
    plt.savefig('SavedData/averageComparison.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()