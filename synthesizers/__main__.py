#from contextlib import redirect_stdout
from pickle import dumps, loads
from sys import stderr, stdin, stdout

while True:
    #print("WORKER: waiting for input", file=stderr)
    header = bytearray()
    header.extend(stdin.buffer.read(8))
    while header[-8:] != b'\x00\x00\x00\x00\x00\x00\x00\x00':
        header.extend(stdin.buffer.read(1))
    #print("WORKER: reading length", file=stderr)
    length = int.from_bytes(stdin.buffer.read(8), byteorder='big')
    #print(f"WORKER: got length {length}", file=stderr)
    inp = stdin.buffer.read(length)
    #print(f"WORKER: got input of length {len(inp)}", file=stderr)
    f, args, kwargs = loads(inp)
    #print(f"WORKER: parsed input to {f} applied to {args}", file=stderr)
    #with redirect_stdout(stderr):
    result = f(*args, **kwargs)
    #print(f"WORKER: result is {result}", file=stderr)
    pickled = dumps(result, protocol=4)
    #print(f"WORKER: pickled result of length {len(pickled)}", file=stderr)
    stdout.buffer.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
    stdout.buffer.write(len(pickled).to_bytes(8, byteorder='big'))
    #print(f"WORKER: wrote length", file=stderr)
    stdout.buffer.write(pickled)
    #print(f"WORKER: wrote pickled result", file=stderr)
    stdout.buffer.flush()
    #print(f"WORKER: flushed buffer", file=stderr)
