from multiprocessing import Process, cpu_count
from multiprocessing.queues import SimpleQueue

class CPUPool(object):
    """Distributes a data ~equally to n_jobs workers with a function to work on it."""

    def __init__(self, n_jobs=0):
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count() / 2
        self.__queue = SimpleQueue()

    def __chunkify(self, data):
        """Chunks a container into ~equal portions to be distributed to workers."""
        n = len(data)
        starts = [(n / self.n_jobs)] * (self.n_jobs-1) + [n - (n/self.n_jobs)*(self.n_jobs-1)]
        i = 0
        for count in starts:
            yield data[i:i+count]
            i += count

    def process(self, data, func, **kwargs):
        """Spawns a process for each chunk executing the given function."""
        self.__pool = []
        for pid, chunk in enumerate(self.__chunkify(data)):
            self.__pool.append(Process(target=lambda i,r,c,f: r.put((i, f(c, **kwargs))),
                                       args=(pid, self.__queue, chunk, kwargs)))
        for p in self.__pool:
            p.start()

    def join(self):
        """Receives the results and the PID's to order the results and return them."""
        results = [None] * self.n_jobs
        for i in range(self.n_jobs):
            pid, res = self.__queue.get()
            results[pid] = res
            self.__pool[pid].join()

        return results
