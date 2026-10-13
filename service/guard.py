"""Production guard: ordinary-user, exclusive ownership, durable stop, no restart."""
import fcntl, hashlib, json, os, signal, subprocess, threading, time, urllib.request
from pathlib import Path
from submission_screen import SubmissionScreen
ROOT=Path(__file__).resolve().parent.parent
MODEL='/path/to/LFM2.5-8B-A1B-Q6_K.gguf'
MODEL_SHA='7ccf57a2d410d8822d1560a1ca10c8318f3e15d6a8f6d42d1903d58e80ea20a6'
ALIAS='lfm2.5-8b-a1b-w9100'
CONFIG_SHA='c8d5145cceb47d01b701a0764817d804a59ddecda67de5dd0d9e34c4aaa4a825'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.pending')
 with t.open('w') as f:json.dump(v,f);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def command(a):return subprocess.check_output(a,stderr=subprocess.STDOUT,timeout=2).decode()
def protected():
 s=command(['systemctl','show','bonsai.service','coolercontrold.service','-p','Id','-p','MainPID','-p','ActiveState','-p','ExecMainStartTimestampMonotonic'])
 assert s.count('ActiveState=active')==2,s
 identity=ROOT/'evidence/protected-config.json'
 if identity.exists():
  expected=json.loads(identity.read_text())
  for path,digest in expected['files'].items():assert sha(path)==digest,('protected config identity changed',path)
  same_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==expected['boot_id']
 else:same_boot=True
 if same_boot:assert 'MainPID=225499\n' in s and 'ExecMainStartTimestampMonotonic=224785644301\n' in s and 'ExecMainStartTimestampMonotonic=42617676254\n' in s,s
 assert sha('/etc/coolercontrol/config.toml')==CONFIG_SHA
 with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as r:assert json.load(r)=={'status':'ok'}
 return s
def telemetry():
 p=Path('/sys/class/drm/card0/device');assert (p/'device').read_text().strip()=='0x67a0'
 t=int(next(p.glob('hwmon/hwmon*/temp1_input')).read_text())/1000
 assert 0<t<80,('temperature boundary',t)
 return {'temp_c':t,'vram_bytes':int((p/'mem_info_vram_used').read_text()),'vram_total':int((p/'mem_info_vram_total').read_text())}
def kernel():return command(['journalctl','-k','-b','--no-pager','-o','short-monotonic','-g','amdgpu|drm|gpu|GPU'])
def faults(s):return [l for l in s.splitlines() if 'amdgpu' in l.lower() and any(w in l.lower() for w in ['timeout','reset','device lost','vm fault','page fault','gpu fault','coredump'])]
def fresh(v,now=None):
 now=time.time_ns() if now is None else now
 return (type(v.get('time_ns')) is int and 0<=now-v['time_ns']<5_000_000_000 and v.get('new_gpu_faults')==0 and v.get('protected_health_failures')==0)
def environment():
 e={k:v for k,v in os.environ.items() if not k.startswith(('LFM_','GGML_','VK_','MESA_')) and k not in ['LD_LIBRARY_PATH','RUSTFLAGS','CARGO_TARGET_DIR']}
 e.update(TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'),MESA_SHADER_CACHE_DIR=str(ROOT/'cache/mesa'),LFM_ATTENTION='coalesced',LFM_Q6_ALIGNED='1')
 return e
class Guard:
 def __init__(self,label,binary):
  self.binary=Path(binary);self.pid=None;self.stopping=False;self.failure=None;self.status={};self.mutex=threading.Lock()
  self.latch=ROOT/'evidence/FAULT-LATCH.json'
  assert not self.latch.exists(),'durable fatal latch refuses startup'
  self.lock=open('/mnt/data/work/.w9100-gpu-owner.lock','a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.run=ROOT/'evidence/raw'/label;self.run.mkdir(parents=True,exist_ok=False)
  self.screen=SubmissionScreen(self.run/'submissions.jsonl');self.intent_since=None
  self.original=protected();self.k0=kernel();self.f0=faults(self.k0)
  preflight=json.loads((ROOT/'evidence/preflight.json').read_text())
  identity=json.loads((ROOT/'evidence/protected-config.json').read_text())
  same_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot_id']
  assert self.f0==(preflight['faults'] if same_boot else []),'GPU faults changed since authorization / new boot has GPU faults'
  for path,digest in preflight['prior_latches'].items():assert sha(path)==digest,'prior latch identity changed'
  (self.run/'kernel-before.log').write_text(self.k0)
  self.trace=open(self.run/'monitor.jsonl','a',buffering=1)
  cert=json.loads((ROOT/'evidence/build-certificate.json').read_text())
  assert cert['pass'] and cert['fence_stop_tests_passed'] and cert['work_bounds_unchanged']
  assert sha(self.binary) in cert['allowed_binaries']
  serving=ROOT/'evidence/serving-source-manifest.json'
  for ref in json.loads(serving.read_text()):assert sha(ROOT/ref['path'])==ref['sha256'],'serving source identity changed'
  for ref in json.loads((ROOT/'evidence/accepted-source-manifest.json').read_text()):assert sha(ROOT/ref['path'])==ref['sha256']
  assert sha(MODEL)==MODEL_SHA
  self.tick()
 def healthy(self):return not self.failure and not self.latch.exists() and fresh(self.status)
 def tick(self):
  t=telemetry();assert protected()==self.original,'protected identity changed'
  k=kernel();assert faults(k)==self.f0,'new GPU fault'
  screen=self.screen.poll()
  if self.screen.intents:
   sequence=self.screen.last_sequence
   if self.intent_since is None or self.intent_since[0]!=sequence:self.intent_since=(sequence,time.monotonic())
   assert time.monotonic()-self.intent_since[1]<2,'incomplete active submission'
  else:self.intent_since=None
  v={'time_ns':time.time_ns(),'new_gpu_faults':0,'protected_health_failures':0,**t,**screen}
  self.status=v;atomic(self.run/'monitor-status.json',v);self.trace.write(json.dumps(v)+'\n')
 def fail(self,reason):
  with self.mutex:
   if self.failure:return
   self.failure=str(reason)
   if not self.latch.exists():atomic(self.latch,{'time_ns':time.time_ns(),'reason':self.failure,'run':str(self.run)})
   if self.pid is not None and self.pid.poll() is None:self.pid.kill()
 def watch(self):
  while not self.stopping and not self.failure:
   try:
    self.tick()
    # stop() may run while tick() is checking protected services. Observe the
    # shutdown flag AFTER polling the child, not only at loop entry.
    if self.pid is not None:
     rc=self.pid.poll()
     if rc is not None and not (self.stopping and rc in (0,-signal.SIGTERM)):
      raise RuntimeError('backend exited: '+str(rc))
   except BaseException as e:self.fail(repr(e));return
   time.sleep(.25)
 def watchdog(self):
  while not self.stopping and not self.failure:
   if not self.healthy():self.fail('monitor missing/stale/unhealthy, including idle');return
   time.sleep(.1)
 def start(self,args,stdin=subprocess.PIPE):
  e=environment();e.update(LFM_SUBMISSION_TRACE=str(self.screen.path),LFM_MONITOR_STATUS=str(self.run/'monitor-status.json'))
  self.err=open(self.run/'backend.stderr','wb')
  self.pid=subprocess.Popen([str(self.binary),*args],stdin=stdin,stdout=subprocess.PIPE,stderr=self.err,env=e,cwd=ROOT)
  atomic(self.run/'launch.json',{'pid':self.pid.pid,'binary':str(self.binary),'sha256':sha(self.binary),'environment':{k:v for k,v in e.items() if k.startswith('LFM_')},'args':args})
  threading.Thread(target=self.watch,daemon=True).start();threading.Thread(target=self.watchdog,daemon=True).start()
  return self.pid
 def stop(self):
  self.stopping=True
  if self.pid is not None and self.pid.poll() is None:
   self.pid.terminate()
   try:self.pid.wait(timeout=3)
   except subprocess.TimeoutExpired:self.pid.kill();self.pid.wait()
  (self.run/'kernel-after.log').write_text(kernel())
  atomic(self.run/'end.json',{'time_ns':time.time_ns(),'failure':self.failure,'status':self.status,'backend_returncode':self.pid.returncode if self.pid else None})
  self.lock.close()
