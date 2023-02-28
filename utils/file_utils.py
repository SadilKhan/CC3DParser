import os,shutil


def get_files(dir):
    """ Get the path of all the files in nested folders"""
    all_files=[]
    for path,subdirs,files in os.walk(dir):
        for name in files:
            all_files.append(os.path.join(path,name))
    if len(all_files)>0:
        return all_files
    else:
        raise FileNotFoundError("No Files found. Please check your directory.")
    
def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)