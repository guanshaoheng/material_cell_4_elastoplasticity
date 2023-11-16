from builtins import range
from builtins import object

__contributor__ = "Lisandro Dalcín"
""" MPIPool wrapped using mpi4py """
# import mpi4py
# mpi4py.rc.threaded = False
from mpi4py import MPI


class MPIPool(object):
    def __init__(self, comm=None, master=0):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.master = master
        self.workers = set(range(self.comm.size))
        self.workers.discard(self.master)

    def is_master(self):
        return self.master == self.comm.rank

    def is_worker(self):
        return self.comm.rank in self.workers

    def map(self, function, iterable):
        assert self.is_master()  # kill the worker threads

        # print('\n\tMaster rank: %d\n\tTotal task num: %d  \n\tWorkers ranks: %s' %
        #       (self.comm.rank, len(iterable), self.workers))
        comm = self.comm
        workerset = self.workers.copy()
        tasklist = [(tid, (function, arg)) for tid, arg in enumerate(iterable)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:
            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task = tasklist.pop()
                # print('worker: %d tag: %d' % (worker, taskid))
                comm.send(task, dest=worker, tag=taskid)  # send the tasks to workers
                # print('Rank: %d tag: %d' % (comm.rank, taskid))

            if tasklist:
                flag = comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)  #
                if not flag:  # if there is no source or no tags
                    continue
            else:
                comm.probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.source
            workerset.add(worker)
            taskid = status.tag
            resultlist[taskid] = result
            pending -= 1

        return resultlist

    def start(self):
        if self.is_master():
            return
        comm = self.comm
        # print('-'*80)
        # print('start: comm.rand=%d' % comm.rank)
        master = self.master
        status = MPI.Status()
        while True:
            # print(comm.rank)
            task = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
            if task is None:
                break
            function, arg = task
            result = function(arg)
            comm.ssend(result, dest=master, tag=status.tag)

    def close(self):
        if not self.is_master():
            return
        for worker in self.workers:
            self.comm.send(None, dest=worker, tag=0)
