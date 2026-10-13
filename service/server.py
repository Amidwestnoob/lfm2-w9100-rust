#!/usr/bin/env python3
"""Bounded loopback HTTP adapter. No inference or tokenization in Python."""
import argparse, http.server, json, queue, signal, socket, threading, time, uuid
from pathlib import Path
from guard import Guard, ROOT, MODEL, ALIAS, sha
MAX_BODY=262144
class Invalid(ValueError):pass
def validate(d,chat):
 if not isinstance(d,dict):raise Invalid('JSON object required')
 try:json.dumps(d,ensure_ascii=False).encode('utf-8')
 except (UnicodeError,RecursionError):raise Invalid('JSON must contain valid Unicode scalars within nesting limits')
 allowed={'model','messages' if chat else 'prompt','max_tokens','stream','temperature','n'}
 if set(d)-allowed:raise Invalid('unsupported options: '+','.join(sorted(set(d)-allowed)))
 if d.get('model')!=ALIAS:raise Invalid('unknown model')
 n=d.get('max_tokens',64)
 if type(n) is not int or not 1<=n<=1024:raise Invalid('max_tokens must be integer 1..1024')
 if type(d.get('stream',False)) is not bool:raise Invalid('stream must be boolean')
 if type(d.get('temperature',0)) not in (int,float) or d.get('temperature',0)!=0:raise Invalid('greedy only: temperature must be zero')
 if type(d.get('n',1)) is not int or d.get('n',1)!=1:raise Invalid('n must be one')
 r={'id':'lfm-'+uuid.uuid4().hex,'n':n}
 if chat:
  ms=d.get('messages')
  if not isinstance(ms,list) or not ms or len(ms)>128:raise Invalid('messages must be nonempty array, at most 128')
  for m in ms:
   if not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) or not m['content']:raise Invalid('only nonempty text system/user/assistant messages supported')
  r['chat']={'messages':ms,'add_generation_prompt':True}
 else:
  p=d.get('prompt')
  if isinstance(p,str) and p:r['text']=p
  elif isinstance(p,list) and p and len(p)<=4096 and all(type(x) is int and 0<=x<2**32 for x in p):r['tokens']=p
  else:raise Invalid('prompt must be nonempty text or token-ID array')
 return r
class Job:
 def __init__(self,r):self.request=r;self.events=queue.Queue(maxsize=1100);self.cancelled=threading.Event()
class Backend:
 def __init__(self,guard):
  self.guard=guard;self.ready=False;self.loads=0;self.count=0;self.active=None;self.waiting=queue.Queue(maxsize=8)
  self.binary_sha=sha(guard.binary);self.log=open(guard.run/'backend.jsonl','a',buffering=1);self.loglock=threading.Lock()
  self.p=guard.start(['--model',MODEL]);threading.Thread(target=self.work,daemon=True).start()
 def read(self):
  line=self.p.stdout.readline()
  if not line:raise RuntimeError('unexpected backend EOF')
  v=json.loads(line)
  with self.loglock:self.log.write(json.dumps({'time_ns':time.time_ns(),'event':v})+'\n')
  return v
 def work(self):
  job=None
  try:
   v=self.read()
   if v['type']!='ready':raise RuntimeError('missing backend readiness')
   self.ready=True;self.loads=v['model_load_count']
   while not self.guard.stopping:
    job=self.waiting.get();self.active=job
    if job.cancelled.is_set():self.active=None;self.waiting.task_done();continue
    if not self.guard.healthy():raise RuntimeError('unhealthy guard before request')
    self.p.stdin.write((json.dumps(job.request,ensure_ascii=False)+'\n').encode());self.p.stdin.flush()
    while True:
     v=self.read();job.events.put_nowait(v)
     if v['type'] in ('result','validation_error'):break
    if v['type']=='result':self.count=v['request_count']
    self.active=None;self.waiting.task_done();job=None
  except BaseException as e:
   self.ready=False
   if not self.guard.stopping:self.guard.fail(repr(e))
   if job:job.events.put_nowait({'type':'fatal','message':str(e)})
 def health(self):
  ok=self.ready and self.guard.healthy() and self.p.poll() is None
  return {'status':'ok' if ok else 'unavailable','model':ALIAS,'ready':ok,'backend_loaded':self.ready and self.p.poll() is None,'backend_sha256':self.binary_sha,'monitor_healthy':self.guard.healthy(),'fatal_latch':self.guard.failure or (str(self.guard.latch) if self.guard.latch.exists() else None),'model_load_count':self.loads,'request_count':self.count,'backend_pid':self.p.pid,'max_context':4096,'max_generation':1024,'decoding':'greedy','queued':self.waiting.qsize(),'active':self.active.request['id'] if self.active else None}
class Server(http.server.ThreadingHTTPServer):
 daemon_threads=True;request_queue_size=16
 def __init__(self,b):self.backend=b;self.slots=threading.BoundedSemaphore(20);super().__init__(('127.0.0.1',18081),Handler)
 def process_request(self,request,address):
  if not self.slots.acquire(False):request.close();return
  try:super().process_request(request,address)
  except BaseException:self.slots.release();raise
 def process_request_thread(self,*a):
  try:super().process_request_thread(*a)
  finally:self.slots.release()
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):super().setup();self.connection.settimeout(15)
 def log_message(self,*args):pass
 def reply(self,status,v):
  body=json.dumps(v,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.send_header('Connection','close');self.end_headers();self.wfile.write(body);self.wfile.flush();self.close_connection=True
 def error(self,status,message):self.reply(status,{'error':{'message':message,'type':'invalid_request_error' if status<500 else 'server_error','code':status}})
 def do_GET(self):
  b=self.server.backend
  if self.path=='/health':
   h=b.health();self.reply(200 if h['ready'] else 503,h)
  elif self.path=='/v1/models':self.reply(200,{'object':'list','data':[{'id':ALIAS,'object':'model','created':0,'owned_by':'local','max_context':4096,'max_generation':1024,'decoding':'greedy'}]})
  else:self.error(404,'unknown endpoint')
 def disconnected(self):
  import select
  if not select.select([self.connection],[],[],0)[0]:return False
  return self.connection.recv(1,socket.MSG_PEEK)==b''
 def next_event(self,job):
  deadline=time.monotonic()+300
  while time.monotonic()<deadline:
   try:return job.events.get(timeout=.2)
   except queue.Empty:
    if self.disconnected():job.cancelled.set();raise BrokenPipeError()
    if self.server.backend.guard.failure:raise RuntimeError('backend fatal stop')
  job.cancelled.set();raise TimeoutError('request deadline; active inference drains before next request')
 def do_POST(self):
  job=None;headers=False
  try:
   chat=self.path=='/v1/chat/completions'
   if not chat and self.path!='/v1/completions':self.error(404,'unknown endpoint');return
   if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1:raise Invalid('one Content-Length required; chunked requests unsupported')
   if self.headers.get_content_type()!='application/json':raise Invalid('Content-Type application/json required')
   try:length=int(self.headers['Content-Length'])
   except ValueError:raise Invalid('invalid Content-Length')
   if not 0<length<=MAX_BODY:raise Invalid('request body must be 1..262144 bytes')
   raw=self.rfile.read(length)
   if len(raw)!=length:raise Invalid('incomplete body')
   def pairs(items):
    d={}
    for k,v in items:
     if k in d:raise Invalid('duplicate JSON key')
     d[k]=v
    return d
   d=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(Invalid('nonfinite JSON')))
   r=validate(d,chat);b=self.server.backend
   if not b.health()['ready']:self.error(503,'backend not ready');return
   job=Job(r)
   try:b.waiting.put_nowait(job)
   except queue.Full:self.error(429,'queue full (eight waiting requests)');return
   v=self.next_event(job)
   if v['type']=='validation_error':self.error(400,v['message']);return
   if v['type']!='accepted':raise RuntimeError('unexpected backend response')
   stream=d.get('stream',False);created=int(time.time())
   base={'id':r['id'],'created':created,'model':ALIAS}
   def sse(v):self.wfile.write(('data: '+(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False))+'\n\n').encode());self.wfile.flush()
   if stream:
    self.send_response(200);self.send_header('Content-Type','text/event-stream; charset=utf-8');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers();headers=True;self.close_connection=True
    if chat:sse({**base,'object':'chat.completion.chunk','choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})
   while True:
    v=self.next_event(job)
    if v['type']=='delta':
     if stream and v['text']:sse({**base,'object':'chat.completion.chunk' if chat else 'text_completion','choices':[{'index':0,**({'delta':{'content':v['text']}} if chat else {'text':v['text']}),'finish_reason':None}]})
    elif v['type']=='result':break
    else:raise RuntimeError('backend fatal response')
   usage={'prompt_tokens':len(v['prompt_tokens']),'completion_tokens':len(v['tokens']),'total_tokens':len(v['prompt_tokens'])+len(v['tokens'])}
   if stream:
    sse({**base,'object':'chat.completion.chunk' if chat else 'text_completion','choices':[{'index':0,**({'delta':{}} if chat else {'text':''}),'finish_reason':v['finish_reason']}],'usage':usage});sse('[DONE]')
   else:self.reply(200,{**base,'object':'chat.completion' if chat else 'text_completion','choices':[{'index':0,**({'message':{'role':'assistant','content':v['text']}} if chat else {'text':v['text']}),'finish_reason':v['finish_reason']}],'usage':usage})
  except (ValueError,UnicodeError,RecursionError) as e:
   if not headers:self.error(400,str(e))
  except (BrokenPipeError,ConnectionResetError,socket.timeout):pass
  except Exception as e:
   if not headers:
    try:self.error(503,str(e))
    except OSError:pass
  finally:
   if job:job.cancelled.set()
   self.close_connection=True

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',default='service-'+str(time.time_ns()));a=ap.parse_args()
 guard=Guard(a.label,ROOT/'target/release/api_stream');server=None
 try:
  backend=Backend(guard);server=Server(backend)
  def stop(*_):guard.stopping=True;threading.Thread(target=server.shutdown,daemon=True).start()
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  server.serve_forever(poll_interval=.2)
 finally:
  if server:server.server_close()
  guard.stop()
if __name__=='__main__':main()
