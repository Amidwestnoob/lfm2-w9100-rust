"""Incremental actual/predictive timing screen, used by every Rust GPU run."""
import json, math
from pathlib import Path
class ScreenStop(RuntimeError): pass
class SubmissionScreen:
    def __init__(self, path):
        self.path=Path(path);self.offset=0;self.pending=b'';self.intents={}
        self.last_sequence=-1;self.completed=0;self.max_gpu_ms=0.0
    @staticmethod
    def number(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ScreenStop('nonfinite/negative timing')
        return value
    def feed(self,row):
        kind=row.get('event');seq=row.get('sequence')
        if kind not in ('submission_intent','gpu_submission'):raise ScreenStop('unknown submission event')
        if isinstance(seq,bool) or not isinstance(seq,int) or seq<0:raise ScreenStop('invalid sequence')
        if kind=='submission_intent':
            if seq!=self.last_sequence+1 or self.intents:raise ScreenStop('duplicate/out-of-order/incomplete submission')
            if self.number(row['predicted_ms'])>80:raise ScreenStop('predicted submission >80ms')
            if not row['commands']:raise ScreenStop('missing commands')
            if any(self.number(c['predicted_ms'])>40 for c in row['commands']):raise ScreenStop('predicted dispatch >40ms')
            self.intents[seq]=row['predicted_ms'];self.last_sequence=seq
        else:
            if seq not in self.intents or row.get('status')!='ok':raise ScreenStop('unmatched/failed submission')
            if row.get('timing_source')!='vulkan_timestamp_queries':raise ScreenStop('missing device timestamp evidence')
            ms=self.number(row['gpu_ms']);self.max_gpu_ms=max(self.max_gpu_ms,ms)
            if ms>=40:raise ScreenStop('actual submission >=40ms: '+str(ms))
            if row['host_wait_complete_monotonic_ns']<row['host_submit_monotonic_ns']:raise ScreenStop('invalid host timing')
            if row['host_wait_complete_monotonic_ns']-row['host_submit_monotonic_ns']>=1_000_000_000:raise ScreenStop('host fence >=1s')
            self.intents.pop(seq);self.completed+=1
    def poll(self,final=False):
        if self.path.exists():
            with self.path.open('rb') as f:
                f.seek(self.offset);data=f.read();self.offset+=len(data)
            parts=(self.pending+data).split(b'\n');self.pending=parts.pop()
            for line in parts:
                if line.strip():self.feed(json.loads(line))
        if final:
            if self.pending.strip():raise ScreenStop('truncated submission trace')
            if self.intents or self.completed==0:raise ScreenStop('incomplete/empty submission trace')
        return {'completed':self.completed,'max_gpu_ms':self.max_gpu_ms}
