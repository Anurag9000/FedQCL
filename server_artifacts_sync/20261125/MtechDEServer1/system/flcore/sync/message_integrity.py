"""Versioned, checksummed simulation messages with bounded duplicate tracking.
Checksums detect accidental corruption, not malicious senders or authentication.
"""
import hashlib,json
from dataclasses import dataclass
def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def packet(round_id,sender,seq,payload):
    envelope={"round":round_id,"sender":sender,"seq":seq,"payload":payload}
    return {**envelope,"digest":hashlib.sha256(canonical(envelope)).hexdigest()}
@dataclass
class Inbox:
    round_id:int
    def __post_init__(self):
        self.seen={}
    def accept(self,message):
        envelope={k:message[k] for k in ("round","sender","seq","payload")}
        if message["digest"]!=hashlib.sha256(canonical(envelope)).hexdigest():return "corrupt",None
        if envelope["round"]!=self.round_id:return "wrong_round",None
        key=(envelope["sender"],envelope["seq"])
        if key in self.seen:
            return ("duplicate" if self.seen[key]==message["digest"] else "conflict"),None
        self.seen[key]=message["digest"]
        return "accepted",envelope["payload"]
