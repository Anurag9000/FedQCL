#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); listing=Path(sys.argv[2]); output=Path(sys.argv[3])
items={}
for raw in listing.read_text().splitlines():
 name=raw.strip().rstrip('/')
 if not name: continue
 folder=root/name
 if not folder.is_dir(): raise SystemExit(f'missing directory: {folder}')
 for path in sorted(folder.rglob('*')):
  if path.is_file():
   h=hashlib.sha256(); size=0
   with path.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):
     h.update(block);size+=len(block)
   items[f'{name}/{path.relative_to(folder).as_posix()}']={'sha256':h.hexdigest(),'bytes':size}
output.write_text(json.dumps(items,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps({'files':len(items),'bytes':sum(x['bytes'] for x in items.values()),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
