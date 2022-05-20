import Const as c
learner_data = {}
teacher_data = {}
total_learnt = {}
iteration = 0
evaluation_iteration = 0


def init_dict():
    data = dict()


def add_student_data(id, content):
    try:
        learner_data[str(id)].append(content)
    except KeyError:
        learner_data[str(id)] = [content]

def add_teacher_data(id, content):
    try:
        teacher_data[str(id)].append(content)
    except KeyError:
        teacher_data[str(id)] = [content]

def print_data():
    print(data)

def save_data(filename=None):
    if filename == None:
        filename = c.SAVE_FILE
    if c.OVERWRITE_FILE == True:
        with open(filename + '_teacher', 'w+') as f:
            for key in teacher_data.keys():
                f.write(str(teacher_data[key]) + '\n')
        with open(filename + '_learner', 'w+') as f:
            for key in learner_data.keys():
                f.write(str(learner_data[key]) + '\n')
        if len(total_learnt)!= 0:
            with open(filename+'_learnt', 'w+') as f:
                for key in total_learnt.keys():
                    f.write(str(total_learnt[key]) + '\n')
    else :
        with open(filename + '_teacher', 'a') as f:
            for key in teacher_data.keys():
                f.write(str(teacher_data[key]) + '\n')
        with open(filename + '_learner', 'a') as f:
            for key in learner_data.keys():
                f.write(str(learner_data[key]) + '\n')
        if len(total_learnt)!= 0:
            with open(filename+'_learnt', 'a') as f:
                for key in total_learnt.keys():
                    f.write(str(total_learnt[key]) + '\n')