import time
def time_computing_decorator(functionnality):
    def time_computing_wrapper(*args, **kwargs):
        start_time = time.time()
        functionnality(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{functionnality.__name__} executed in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    return time_computing_wrapper