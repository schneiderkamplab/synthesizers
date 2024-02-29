from itertools import zip_longest
from pickle import dumps, loads
import subprocess
import sys

def init_process(module_name):
    return subprocess.Popen([sys.executable, '-m', module_name], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

class SubprocessPool:

    def __init__(self, n_workers, module_name):
        self.n_workers = n_workers
        self.module_name = module_name
        self.workers = [init_process(module_name) for _ in range(n_workers)]
        self.active_workers = []
        self.next_worker = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self

    def __del__(self):
        for worker in self.workers:
            worker.stdin.close()
            worker.stdout.close()
            worker.terminate()

    def map(self, func, argss, kwargss=[]):
        for args, kwargs in zip_longest(argss, kwargss, fillvalue={}):
            if len(self.active_workers) == self.n_workers:
                worker = self.active_workers.pop(0)
                header = bytearray()
                header.extend(worker.stdout.read(8))
                while header[-8:] != b'\x00\x00\x00\x00\x00\x00\x00\x00':
                    header.extend(worker.stdout.read(1))
                length = int.from_bytes(worker.stdout.read(8), byteorder='big')
                result = worker.stdout.read(length)
                yield loads(result)
            worker = self.workers[self.next_worker]
            self.next_worker = (self.next_worker + 1) % self.n_workers
            self.active_workers.append(worker)
            pickled = dumps((func, args, kwargs))
            worker.stdin.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            worker.stdin.write(len(pickled).to_bytes(8, byteorder='big'))
            worker.stdin.write(pickled)
            worker.stdin.flush()
        while self.active_workers:
            worker = self.active_workers.pop(0)
            header = bytearray()
            header.extend(worker.stdout.read(8))
            while header[-8:] != b'\x00\x00\x00\x00\x00\x00\x00\x00':
                header.extend(worker.stdout.read(1))
            length = int.from_bytes(worker.stdout.read(8), byteorder='big')
            result = worker.stdout.read(length)
            yield loads(result)
