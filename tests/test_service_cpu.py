import io,json,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'service'))
import guard,server
from submission_screen import SubmissionScreen,ScreenStop
class CPU(unittest.TestCase):
 def test_validation(self):
  good={'model':guard.ALIAS,'prompt':'hello','temperature':0,'max_tokens':1}
  self.assertEqual(server.validate(good,False)['text'],'hello')
  for update in [{'temperature':1},{'top_p':1},{'tools':[]},{'max_tokens':True},{'max_tokens':1025},{'max_tokens':0},{'stream':1},{'prompt':[]},{'prompt':[-1]},{'prompt':[True]},{'n':2},{'model':'other'}]:
   with self.assertRaises(server.Invalid):server.validate({**good,**update},False)
 def test_unicode_scalar_validation(self):
  for d,chat in [({'model':guard.ALIAS,'prompt':'\ud800'},False),({'model':guard.ALIAS,'messages':[{'role':'user','content':'\udfff'}]},True),({'model':guard.ALIAS,'prompt':'ok','\ud800':1},False)]:
   with self.assertRaises(server.Invalid):server.validate(d,chat)
 def test_monitor_missing_delayed_future_unhealthy(self):
  n=time.time_ns();v={'time_ns':n,'new_gpu_faults':0,'protected_health_failures':0}
  self.assertTrue(guard.fresh(v,n))
  for x in [{},{**v,'time_ns':n-5_000_000_000},{**v,'time_ns':n+1},{**v,'new_gpu_faults':1},{**v,'protected_health_failures':1}]:self.assertFalse(guard.fresh(x,n))
 def test_watchdog_kills_even_idle(self):
  g=object.__new__(guard.Guard);g.stopping=False;g.failure=None;g.healthy=lambda:False
  reasons=[]
  def fail(s):reasons.append(s);g.failure=s
  g.fail=fail;g.watchdog();self.assertEqual(len(reasons),1)
 def test_idle_screen_and_active_incomplete(self):
  s=SubmissionScreen('/nonexistent-cpu-fixture');self.assertEqual(s.poll()['completed'],0)
  s.feed({'event':'submission_intent','sequence':0,'predicted_ms':1,'commands':[{'predicted_ms':1}]})
  self.assertEqual(len(s.intents),1)
  with self.assertRaises(ScreenStop):s.poll(final=True)
 def test_measured_and_predictions(self):
  for ms in [40,41,float('nan')]:
   s=SubmissionScreen('/nonexistent');s.feed({'event':'submission_intent','sequence':0,'predicted_ms':1,'commands':[{'predicted_ms':1}]})
   with self.assertRaises(ScreenStop):s.feed({'event':'gpu_submission','sequence':0,'status':'ok','timing_source':'vulkan_timestamp_queries','gpu_ms':ms,'host_wait_complete_monotonic_ns':2,'host_submit_monotonic_ns':1})
  for total,dispatch in [(81,1),(1,41)]:
   with self.assertRaises(ScreenStop):SubmissionScreen('/nonexistent').feed({'event':'submission_intent','sequence':0,'predicted_ms':total,'commands':[{'predicted_ms':dispatch}]})
 def test_sticky_fatal_no_restart_or_latch_clear(self):
  import tempfile,threading
  from unittest.mock import Mock
  with __import__('contextlib').nullcontext(tempfile.mkdtemp(dir=guard.ROOT/'tmp')) as d:
   root=Path(d);(root/'evidence').mkdir()
   g=object.__new__(guard.Guard);g.failure=None;g.mutex=threading.Lock();g.latch=root/'evidence/FAULT-LATCH.json';g.run=root;g.pid=Mock();g.pid.poll.return_value=None
   g.fail('synthetic monitor loss');first=g.latch.read_bytes();g.fail('second failure');self.assertEqual(g.latch.read_bytes(),first);g.pid.kill.assert_called_once()
   with patch.object(guard,'ROOT',root):
    with self.assertRaisesRegex(AssertionError,'durable fatal latch'):guard.Guard('never-starts',root/'binary')
 def test_idle_and_incomplete_monitor_tick(self):
  import tempfile
  from unittest.mock import Mock
  with __import__('contextlib').nullcontext(tempfile.mkdtemp(dir=guard.ROOT/'tmp')) as d:
   g=object.__new__(guard.Guard);g.run=Path(d);g.original='protected';g.f0=[];g.trace=io.StringIO();g.intent_since=None;g.screen=SubmissionScreen(g.run/'trace')
   with patch.object(guard,'telemetry',return_value={'temp_c':40}),patch.object(guard,'protected',return_value='protected'),patch.object(guard,'kernel',return_value=''):
    g.tick();g.tick();self.assertTrue(guard.fresh(g.status));self.assertEqual(g.status['completed'],0)
    g.screen.feed({'event':'submission_intent','sequence':0,'predicted_ms':1,'commands':[{'predicted_ms':1}]});g.tick()
    g.intent_since=(0,time.monotonic()-3)
    with self.assertRaisesRegex(AssertionError,'incomplete active'):g.tick()
 def test_administrative_stop_during_monitor_tick_not_fatal(self):
  from unittest.mock import Mock
  g=object.__new__(guard.Guard);g.stopping=False;g.failure=None;g.pid=Mock();g.pid.poll.return_value=-15;g.fail=Mock()
  def tick():g.stopping=True
  g.tick=tick
  with patch.object(guard.time,'sleep'):g.watch()
  g.fail.assert_not_called()
 def test_unexpected_exit_still_fatal(self):
  from unittest.mock import Mock
  for rc in [-15,1]:
   g=object.__new__(guard.Guard);g.stopping=False;g.failure=None;g.pid=Mock();g.pid.poll.return_value=rc;g.tick=lambda:None;g.fail=Mock()
   g.watch();g.fail.assert_called_once()
 def test_safe_validation_not_fatal(self):
  p=guard.ROOT/'evidence/FAULT-LATCH.json';before=p.read_bytes() if p.exists() else None
  with self.assertRaises(server.Invalid):server.validate({'model':guard.ALIAS,'prompt':'','max_tokens':1},False)
  self.assertEqual(p.read_bytes() if p.exists() else None,before)
if __name__=='__main__':unittest.main(verbosity=2)
