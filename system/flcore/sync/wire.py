"""Versioned compact wire envelope with canonical JSON metadata and binary vectors.
Counts use little-endian uint32; real reports use float64 to preserve policy decisions.
The wire representation is decoded before exposing data to a recipient.
"""
import json
import struct
import math

MAGIC=b"FQC1"
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")

def quantize_report(report,mode):
    from dataclasses import replace
    if report is None or report.metric=="tv" or mode not in ("fp16","int8"):
        return report,0.
    values=report.counts
    if mode=="fp16":
        decoded=tuple(struct.unpack("<e",struct.pack("<e",max(-65504.,min(65504.,v))))[0] for v in values)
        scale=0.
    else:
        scale=max((abs(x) for x in values),default=0.)/127 or 1.
        decoded=tuple(round(max(-127,min(127,x/scale)))*scale for x in values)
    error=math.sqrt(sum((x-y)**2 for x,y in zip(values,decoded)))
    return replace(report,counts=decoded,quant_scale=scale),error

def pack(value,mode="binary"):
    blocks=[]
    def visit(x):
        if isinstance(x,dict):
            if {"counts","labels","schema","metric"}.issubset(x):
                values=x["counts"]
                integer=x["metric"]=="tv"
                if integer and any(type(v) is not int or not 0<=v<2**32 for v in values):
                    raise ValueError("histogram count out of uint32 range")
                if not integer and any(not math.isfinite(v) for v in values):
                    raise ValueError("nonfinite vector")
                fmt="I" if integer else "e" if mode=="fp16" else "b" if mode=="int8" else "d"
                scale=x.get("quant_scale",0.)
                if fmt=="b":
                    if not math.isfinite(scale) or scale<=0:raise ValueError("int8 report requires prequantization")
                    values=[max(-127,min(127,round(v/scale))) for v in values]
                block=struct.pack("<"+fmt*len(values),*values)
                marker={"__fqcl_vector__":len(blocks),"dtype":fmt,"count":len(values)}
                if fmt=="b":marker["scale"]=scale
                blocks.append(block)
                return {k:marker if k=="counts" else visit(v) for k,v in x.items()}
            return {str(k):visit(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [visit(v) for v in x]
        return x
    metadata=canonical(visit(value))
    return MAGIC+struct.pack("<II",len(metadata),len(blocks))+metadata+b"".join(struct.pack("<I",len(b))+b for b in blocks)

def unpack(wire):
    if wire[:4]!=MAGIC or len(wire)<12:raise ValueError("invalid wire magic")
    size,n=struct.unpack_from("<II",wire,4);cursor=12+size
    if cursor>len(wire):raise ValueError("truncated metadata")
    metadata=json.loads(wire[12:cursor])
    blocks=[]
    for _ in range(n):
        if cursor+4>len(wire):raise ValueError("truncated block length")
        m=struct.unpack_from("<I",wire,cursor)[0];cursor+=4
        if cursor+m>len(wire):raise ValueError("truncated block")
        blocks.append(wire[cursor:cursor+m]);cursor+=m
    if cursor!=len(wire):raise ValueError("trailing bytes")
    def visit(x):
        if isinstance(x,dict):
            if "__fqcl_vector__" in x:
                fmt=x["dtype"];count=x["count"];idx=x["__fqcl_vector__"]
                if fmt not in ("I","d","e","b") or count<0 or not 0<=idx<len(blocks):raise ValueError("invalid vector descriptor")
                if len(blocks[idx])!=count*struct.calcsize("<"+fmt):raise ValueError("invalid vector length")
                values=list(struct.unpack("<"+fmt*count,blocks[idx]))
                if fmt=="b":
                    scale=x.get("scale",0.)
                    if not math.isfinite(scale) or scale<=0:raise ValueError("invalid scale")
                    values=[v*scale for v in values]
                return values
            return {k:visit(v) for k,v in x.items()}
        if isinstance(x,list):return [visit(v) for v in x]
        return x
    return visit(metadata)

def binary_vector_bytes(wire):
    """Exact bytes in binary vector blocks, excluding framing."""
    if wire[:4]!=MAGIC or len(wire)<12:raise ValueError("invalid wire")
    size,n=struct.unpack_from("<II",wire,4);cursor=12+size
    total=0
    for _ in range(n):
        m=struct.unpack_from("<I",wire,cursor)[0];cursor+=4+m;total+=m
    if cursor!=len(wire):raise ValueError("invalid wire length")
    return total
