import matplotlib.pyplot as plt
import numpy as np
import Const as c

learner_data = None
teacher_data = None
if __name__ == '__main__':
    total_iterations = 0
    with open("Data.values", 'r') as f:
        for i,each in enumerate(f.readlines()):
            lst = each.strip('[]\n').split(', ')
            if learner_data == None:
                learner_data = [[] for _ in range(len(lst))]
                teacher_data = [[] for _ in range(len(lst))]
                total_iterations = len(lst)
            for ite, score in enumerate(lst):
                if i < c.NB_LEARNER:
                    learner_data[ite].append(float(score))
                else :
                    teacher_data[ite].append(float(score))
    print(np.sum(learner_data, axis=1))
    plt.plot(list(range(total_iterations)), learner_data, label = 'learner')
    plt.plot(list(range(total_iterations)),[np.mean(x) for x in teacher_data])
    plt.show()
