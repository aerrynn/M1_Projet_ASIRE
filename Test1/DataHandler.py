import Const as c


data = {}
iteration = 0
evaluation_iteration = 0


def init_dict():
    data = dict()


def add_data(id, content):
    try:
        data[id].append(content)
    except KeyError:
        data[id] = [content]


def save_data(filename=None):
    if filename == None:
        filename = c.SAVE_FILE
    with open(filename, 'w+') as f:
        for each in data.keys():
            f.write(str(data[each])+'\n')
