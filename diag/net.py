import socket
import sys
import time

h = socket.gethostname()
print(sys.executable, sys.version.split()[0], "hostname:", h, flush=True)
for name in [h, "localhost"]:
    t = time.time()
    try:
        infos = socket.getaddrinfo(name, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        print(f"getaddrinfo({name}) ok in {time.time() - t:.2f}s:", sorted({i[4][0] for i in infos}), flush=True)
    except Exception as e:
        print(f"getaddrinfo({name}) FAILED in {time.time() - t:.2f}s:", e, flush=True)
        infos = []
    for fam, _, _, _, sa in infos:
        ip = sa[0]
        s = socket.socket(fam, socket.SOCK_STREAM)
        t = time.time()
        try:
            s.bind((ip, 0))
            s.listen(1)
            port = s.getsockname()[1]
            c = socket.socket(fam, socket.SOCK_STREAM)
            c.settimeout(10)
            c.connect((ip, port))
            print(f"  {ip} bind+connect OK {time.time() - t:.2f}s", flush=True)
        except Exception as e:
            print(f"  {ip} FAILED {time.time() - t:.2f}s: {type(e).__name__} {e}", flush=True)

for label, fn in [
    ("getfqdn()", socket.getfqdn),
    ("gethostbyname(hostname)", lambda: socket.gethostbyname(h)),
    ("gethostbyaddr(127.0.0.1)", lambda: socket.gethostbyaddr("127.0.0.1")),
]:
    t = time.time()
    try:
        r = fn()
        print(f"{label} -> {r} in {time.time() - t:.2f}s", flush=True)
    except Exception as e:
        print(f"{label} FAILED in {time.time() - t:.2f}s: {e}", flush=True)
