import matplotlib.pyplot as plt
import numpy as np
import Const as c
import json

learner_data = None
teacher_data = None


def draw_plot(ax, data, offset, edge_color, fill_color, bool):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions=pos, widths=0.3,
                    patch_artist=True, showfliers=bool)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def main1():
    global ax
    learner_data = None
    teacher_data = None
    total_iterations = 0
    with open(c.SAVE_FILE + '_2_teacher', 'r') as f:
        for i, each in enumerate(f.readlines()):
            lst = each.strip('[]\n').split(', ')
            if teacher_data == None:
                teacher_data = [[] for _ in range(len(lst))]
                total_iterations = len(lst)
            for ite, score in enumerate(lst):
                teacher_data[ite].append(float(score))
    # with open(c.SAVE_FILE + '_2_learner', 'r') as f:
    #     for i, each in enumerate(f.readlines()):
    #         lst = each.strip('[]\n').split(', ')
    #         if learner_data == None:
    #             learner_data = [[] for _ in range(len(lst))]
    #             total_iterations = len(lst)
    #         for ite, score in enumerate(lst):
    #             learner_data[ite].append(float(score))
    # fig1, ax1 = plt.subplots()
    # sub_learner_data = np.array(learner_data)
    # print(sub_learner_data.shape)
    # draw_plot(ax1, np.array(
    #     sub_learner_data[::2500]).T, 0, 'black', 'white', False)
    # plt.xticks(range(0, 96, 16), [0, 20000, 40000, 60000, 80000, 100000])
    # plt.savefig('SavedData/learner_boxplot1.png', bbox_inches='tight')
    # fig2, ax2 = plt.subplots()
    # sub_teacher_data = np.array(teacher_data)
    # print(sub_teacher_data.shape)
    # draw_plot(ax2, np.array(
    #     sub_teacher_data[::2500]).T, 0, 'blue', 'white', False)

    # plt.xticks(range(0, 96, 16), [0, 20000, 40000, 60000, 80000, 100000])
    # plt.savefig('SavedData/teach_boxplot1.png', bbox_inches='tight')
    # plt.show()
    # length = len(learner_data[::c.EVALUATION_TIME])
    # plt.plot(np.linspace(0, 100000, 42), [np.mean(
    #     x) for x in learner_data[::8*c.EVALUATION_TIME]], color='red', label='Learner Fitness, D = 2')
    subTD = [np.mean(
        x) for x in teacher_data[::8*c.EVALUATION_TIME]]
    n = np.mean(subTD)
    subTD[0] = n
    plt.plot(np.linspace(0, 200000, 84), subTD,
             color='black', label='Teacher Fitness')
    print(len(lst))
    for filename, color, disc in [('NewData/Disc_2_learner', '#ffa600', '2'), ('NewData/Disc_4_learner', '#ef5675', '4'), ('NewData/Disc_8_learner', '#7a5195', '8'), ('NewData/Disc_16_learner', '#003f5c', '16')]:
        with open(filename, 'r') as f:
            learner_data = [[] for _ in range(400000)]
            for i, each in enumerate(f.readlines()):
                lst = each.strip('[]\n').split(', ')
                if len(lst) < 400000 : 
                    print("pass")
                    continue
                if learner_data == None:
                    learner_data = [[] for _ in range(len(lst))]
                    total_iterations = len(lst)
                for ite, score in enumerate(lst):
                    learner_data[ite].append(float(score))
        plt.plot(np.linspace(0, 200000, 84), [np.mean(
        x) for x in learner_data[::8*c.EVALUATION_TIME]], color=color, label=f'Learner Fitness, D = {disc}')
    plt.legend(loc='lower right')
    plt.savefig('SavedData/averageComparisons.png')
    plt.show()
    plt.close()

def main2():
    for i in (2,4,8,16):
        data = []
        with open(c.SAVE_FILE + '_' + str(i) +'_learnt', 'r') as f:
            for each in f.readlines():
                lst = each.strip('[]\n').split(', ')
                data.append(float(lst[-1]))
        plt.boxplot(data)
        plt.xticks([1], [str(i)])
        plt.title(f"Average learnt behaviour out of the {i**10:,} possibles")
        plt.savefig(f'SavedData/averageLearntBehaviourForD{str(i)}.png')
        plt.show()


if __name__ == '__main__':
    main1()
