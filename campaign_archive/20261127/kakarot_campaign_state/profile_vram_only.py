import concurrent.futures,importlib.util,json,pathlib,sqlite3,subprocess,datetime,sys
S=pathlib.Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('d',S/'dispatch_jobs.py');d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)
GPU='nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits'
APP='nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits'
c=sqlite3.connect(S/'fleet_state.sqlite3');active={}
for host,jid,pid in c.execute("select a.host,j.job_id,a.pid from attempts a join jobs j on j.job_id=a.job_id where a.state='RUNNING'"):
 if pid is not None:active[host,str(pid)]=jid
c.close()
def fetch(h):
 if h=='kakarot':
  g=subprocess.run(GPU.split(),capture_output=True,text=True,check=True).stdout
  a=subprocess.run(APP.split(),capture_output=True,text=True).stdout
 else:
  rc,g,e=d.invoke(h,d.ssh_argv(h,GPU),timeout=60)
  if rc not in (0,None):raise RuntimeError(h+' GPU query failed '+e[-200:])
  rc,a,e=d.invoke(h,d.ssh_argv(h,APP),timeout=60)
  if rc not in (0,None):raise RuntimeError(h+' app VRAM query failed '+e[-200:])
 rows=[]
 for line in g.replace('\r','').splitlines():
  f=[x.strip() for x in line.split(',')]
  if len(f)==6 and f[0].isdigit():rows.append({'index':int(f[0]),'uuid':f[1],'name':f[2],'total_mib':int(f[3]),'used_mib':int(f[4]),'free_mib':int(f[5])})
 apps=[]
 for line in a.replace('\r','').splitlines():
  f=[x.strip() for x in line.split(',')]
  if len(f)==3 and f[0].isdigit() and f[2].isdigit():apps.append({'pid':int(f[0]),'uuid':f[1],'used_mib':int(f[2]),'job_id':active.get((h,f[0]))})
 return h,rows,apps
hosts=['kakarot','mtech','v100','l40','h100'];allrows=[];host_errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as ex:
 futures={ex.submit(fetch,h):h for h in hosts}
 for future in concurrent.futures.as_completed(futures):
  expected_host=futures[future]
  try:
   h,rows,apps=future.result()
  except Exception as exc:
   # A temporary route/authentication failure for one host must not prevent
   # fresh VRAM readings from reachable GPUs. Missing devices remain unknown;
   # fleet admission will refuse to declare the fleet saturated without them.
   host_errors.append({'host':expected_host,'error':repr(exc)})
   continue
  for r in rows:
   ps=[p for p in apps if p['uuid']==r['uuid']]
   r.update(host=h,processes=ps)
   args=['python3',str(S/'fleet_state.py'),'snapshot','--host',h,'--uuid',r['uuid'],'--index',str(r['index']),'--name',r['name'],'--total',str(r['total_mib']*1048576),'--used',str(r['used_mib']*1048576),'--free',str(r['free_mib']*1048576),'--processes',json.dumps(ps,separators=(',',':')),'--context','fleet-vram-process-memory-profile']
   subprocess.run(args,check=True,stdout=subprocess.DEVNULL)
   allrows.append(r)
bcmd="hostname; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits; else echo NO_NVIDIA_SMI; fi"
try:
 bp=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','b550',bcmd],capture_output=True,text=True,timeout=30)
 b550={'reachable':bp.returncode==0,'returncode':bp.returncode,'raw_vram_query':bp.stdout.strip(),
       'stderr':bp.stderr[-500:] if bp.returncode else '',
       'gpu_present':bool([x for x in bp.stdout.splitlines()[1:] if x.strip() and x.strip()!='NO_NVIDIA_SMI'])}
except Exception as e:
 b550={'reachable':False,'returncode':None,'raw_vram_query':'','stderr':repr(e),'gpu_present':None}
out={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'measurement_scope':'VRAM only; no GPU utilization queried','devices':allrows,'unreachable_hosts':host_errors,'b550':b550}
p=S/'vram_process_profile_latest.json';p.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
for r in allrows:print(json.dumps({'host':r['host'],'gpu':r['index'],'name':r['name'],'used_mib':r['used_mib'],'free_mib':r['free_mib'],'processes':r['processes']}))
for e in host_errors:print(json.dumps({'host':e['host'],'state':'HOST_UNREACHABLE','error':e['error']}),file=sys.stderr)
