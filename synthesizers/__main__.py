from contextlib import contextmanager
import os
from pickle import dumps, loads
from sys import stderr, stdin, stdout

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(stdout, to=os.devnull):
    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

while True:
    print("WORKER: waiting for input", file=stderr)
    header = bytearray()
    header.extend(stdin.buffer.read(8))
    while header[-8:] != b'\x00\x00\x00\x00\x00\x00\x00\x00':
        header.extend(stdin.buffer.read(1))
    print("WORKER: reading length", file=stderr)
    length = int.from_bytes(stdin.buffer.read(8), byteorder='big')
    print(f"WORKER: got length {length}", file=stderr)
    inp = stdin.buffer.read(length)
    print(f"WORKER: got input of length {len(inp)}", file=stderr)
    f, args = loads(inp)
    print(f"WORKER: parsed input to {f} applied to {args}", file=stderr)
    with stdout_redirected(stdout, to=stderr):
        result = f(*args)
    print(f"WORKER: result is {result}", file=stderr)
    pickled = dumps(result, protocol=4)
    print(f"WORKER: pickled result of length {len(pickled)}", file=stderr)
    stdout.buffer.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
    stdout.buffer.write(len(pickled).to_bytes(8, byteorder='big'))
    print(f"WORKER: wrote length", file=stderr)
    stdout.buffer.write(pickled)
    print(f"WORKER: wrote pickled result", file=stderr)
    stdout.buffer.flush()
    print(f"WORKER: flushed buffer", file=stderr)
