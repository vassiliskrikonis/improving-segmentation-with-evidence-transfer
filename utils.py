import datetime
from os.path import join


def get_new_logdir(prefix='', root_dir='.'):
    now = datetime.datetime.now()
    now = '{:%Y-%m-%d %H:%M:%S}'.format(now)
    return join(root_dir, prefix, now)
