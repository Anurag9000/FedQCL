#!/usr/bin/env python3
"""Real TCP loopback application-payload measurement; not physical WAN/link bytes."""
import argparse,hashlib,json,socket,threading,time,struct
from pathlib import Path

def benchmark(size,repeats):
    data=b"\x00"*min(size,1048576)
    listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    listener.bind(("127.0.0.1",0));listener.listen(1);listener.settimeout(30)
    counts=[];errors=[]
    def receiver():
        try:
            with listener.accept()[0] as conn:
                conn.settimeout(30)
                for _ in range(repeats):
                    header=b""
                    while len(header)<8:
                        chunk=conn.recv(8-len(header))
                        if not chunk:raise EOFError("missing frame")
                        header+=chunk
                    length=struct.unpack("!Q",header)[0];remaining=length;digest=hashlib.sha256()
                    while remaining:
                        chunk=conn.recv(min(remaining,1048576))
                        if not chunk:raise EOFError("incomplete payload")
                        remaining-=len(chunk);digest.update(chunk)
                    counts.append(length);conn.sendall(digest.digest())
        except Exception as e:errors.append(repr(e))
        finally:listener.close()
    thread=threading.Thread(target=receiver);thread.start();times=[]
    try:
        with socket.create_connection(listener.getsockname(),timeout=30) as conn:
            for _ in range(repeats):
                start=time.perf_counter();conn.sendall(struct.pack("!Q",size));remaining=size;digest=hashlib.sha256()
                while remaining:
                    block=data[:min(remaining,len(data))];conn.sendall(block);remaining-=len(block);digest.update(block)
                ack=b""
                while len(ack)<32:
                    chunk=conn.recv(32-len(ack))
                    if not chunk:raise EOFError("missing acknowledgement")
                    ack+=chunk
                if ack!=digest.digest():raise RuntimeError("payload digest mismatch")
                times.append(time.perf_counter()-start)
    finally:thread.join(30)
    if errors or len(counts)!=repeats:raise RuntimeError(str(errors))
    return {"payload_bytes":size,"repeats":repeats,"application_bytes_sent":repeats*(size+8+32),
            "received_payload_bytes":sum(counts),"roundtrip_seconds":times,"scope":"TCP loopback application frames only"}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--size",type=int,default=1024);p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.size<1 or a.repeats<1:raise ValueError("invalid workload")
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(benchmark(a.size,a.repeats),indent=2))
