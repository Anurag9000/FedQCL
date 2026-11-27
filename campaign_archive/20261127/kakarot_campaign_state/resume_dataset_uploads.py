import concurrent.futures,hashlib,importlib.util,pathlib,shlex
S=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('d',S/'dispatch_jobs.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
src=pathlib.Path('/tmp/fedqcl-compact-primary-datasets-65ced619.tar')
want='be89517591623aeeb63d9f8a477cc16c12f64727c240ee6c9a3d39c6dee65817'
assert hashlib.sha256(src.read_bytes()).hexdigest()==want

def upload(h):
 host=d.HOSTS[h]
 target=(f"{host['user']}@{host['ip']}" if host.get('ip') else host['alias'])+':/tmp/fedqcl-compact-primary-datasets-65ced619.tar'
 transport='ssh -J b550 -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new'
 cmd=['rsync','-rt','--partial','--append-verify','--info=stats2','-e',transport,str(src),target]
 rc,out,err=d.invoke(h,cmd,timeout=7200)
 if rc not in (0,None):raise RuntimeError(f'{h} resumable rsync failed rc={rc}: {(err or out)[-500:]}')
 rc,out,err=d.invoke(h,d.ssh_argv(h,'sha256sum /tmp/fedqcl-compact-primary-datasets-65ced619.tar'),timeout=300)
 if rc not in (0,None) or want not in out:raise RuntimeError(f'{h} transferred archive checksum mismatch')
 return h, want, out.strip().splitlines()[-1]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(upload,['h100','mtech','v100']):print(result,flush=True)
