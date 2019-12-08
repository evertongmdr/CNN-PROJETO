import os
def order_by(elm):
    return elm[1].name

def get_paths(files):
    files = sorted(enumerate(files),key = order_by)
    f = []
    for n,direntry in files:
        f.append(direntry.path)
    return f

def get_name(files):
    files = sorted(enumerate(files), key=order_by)
    f = []
    for n,direntry in files:
        f.append(direntry.name)
    return f

def qtd_files(set_path):

    qtd = 0
    for n,path in enumerate(os.scandir(set_path)):
        qtd += len(get_name(os.scandir(path.path)))
    return qtd