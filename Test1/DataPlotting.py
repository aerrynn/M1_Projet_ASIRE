import matplotlib.pyplot as plt
import numpy as np
import Const as c

learner_data = None
teacher_data = None


def draw_plot(data, offset, edge_color, fill_color):
    pos = (np.linspace(0, 100000, data.shape[1]))
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


if __name__ == '__main__':
    total_iterations = 0
    with open(c.SAVE_FILE, 'r') as f:
        for i, each in enumerate(f.readlines()):
            lst = each.strip('[]\n').split(', ')
            if learner_data == None:
                learner_data = [[] for _ in range(len(lst))]
                teacher_data = [[] for _ in range(len(lst))]
                total_iterations = len(lst)
            for ite, score in enumerate(lst):
                if i < c.NB_LEARNER:
                    learner_data[ite].append(float(score))
                else:
                    teacher_data[ite].append(float(score))
    fig, ax = plt.subplots()
    # plt.boxplot(teacher_data[-1], list(range(total_iterations)))
    # print(len(learner_data[::10000]))
    sub_learner_data = learner_data[:40000]
    sub_teacher_data = teacher_data[:600]
    draw_plot(np.array(sub_learner_data[::600]).T, -.2, 'black', 'red')
    # ax.set_xticks(np.linspace(0, 100000, 10))
    # print(np.array(sub_teacher_data[::10000]).shape)
    # draw_plot(np.array(sub_teacher_data[::10]).T, +.2, 'blue', 'cyan')

    # plt.xticks(range(5))
    # plt.savefig('boxplot.png', bbox_inches='tight')
    plt.show()
    length = len(learner_data[::4*c.EVALUATION_TIME])
    plt.plot(np.linspace(0, 100000, length), [np.mean(
        x) for x in learner_data[::4*c.EVALUATION_TIME]])
    plt.plot(np.linspace(0, 100000, length), [np.mean(
        x) for x in teacher_data[::4*c.EVALUATION_TIME]])
    plt.show()
    plt.close()
