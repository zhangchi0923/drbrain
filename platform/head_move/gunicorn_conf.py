# gunicorn configuration

import multiprocessing
import gevent.monkey
gevent.monkey.patch_all()

bind = "127.0.0.1:8101"
workers = multiprocessing.cpu_count()*2 + 1
threads = 2

worker_class = "gevent"
worker_connections = 100

pidfile = 'gunicorn-log/gunicorn.pid'
accesslog = 'gunicorn-log/access.log'
errorlog = 'gunicorn-log/gunicorn.log'

log_level = "debug"
