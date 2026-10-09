import unittest
import json
import score_v2 as m
class Tests(unittest.TestCase):
 def response(self,**changes):
  r=dict(scope='btc_general',label='positive',horizon='forward',evidence_ids=['H'],reason='Explicit BTC outlook.');r.update(changes)
  return dict(done=True,response=json.dumps(r))
 def test_verbatim_evidence(self):
  text={'H':'Bitcoin’s outlook improves — analyst','S':'Text'}
  self.assertEqual(m.validate(self.response(),text)['evidence'][0]['text'],text['H'])
 def test_missing_segment_rejected(self):
  with self.assertRaises(ValueError): m.validate(self.response(evidence_ids=['S']),{'H':'title'})
 def test_excluded_positive_rejected(self):
  with self.assertRaises(ValueError): m.validate(self.response(scope='relative_only'),{'H':'title'})
 def test_truncated_output_rejected(self):
  r=self.response();r['done_reason']='length'
  with self.assertRaises(ValueError): m.validate(r,{'H':'title'})
 def test_mixed_both_segments(self):
  r=m.validate(self.response(label='mixed',evidence_ids=['H','S']),{'H':'Falls first','S':'Rallies later'})
  self.assertEqual(len(r['evidence']),2)
 def test_duplicate_ids_rejected(self):
  with self.assertRaises(ValueError): m.validate(self.response(evidence_ids=['H','H']),{'H':'title'})
if __name__=='__main__':unittest.main()
