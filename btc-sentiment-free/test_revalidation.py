import json,unittest
import btc_sentiment as b
import revalidate_evidence as r

class RevalidationTests(unittest.TestCase):
 def test_safe_normalization_and_spans(self):
  text='BTC’s  Price\nRises'
  result={'scope':'btc_general','label':'positive','evidence':"btc's price rises",'reason':'rise'}
  fixed=r.repair(result,text)
  self.assertEqual(fixed['evidence'],text)
  self.assertEqual(fixed['evidence_audit']['model_evidence'],result['evidence'])
  for evidence in ['BTC ... Rises','BTC falls', 'BTC Price Rises']:
   with self.assertRaises(ValueError):r.match_span(text,evidence)
 def test_replay_idempotent_preserves_original_and_unattempted(self):
  c=b.connect(':memory:')
  c.execute('CREATE TABLE score_attempts(id INTEGER PRIMARY KEY,at TEXT,run TEXT,url TEXT,text_hash TEXT,request TEXT,response TEXT,error TEXT)')
  c.execute('INSERT INTO runs VALUES(?,?)',('old','{}'))
  for u in ['a','b','c']:
   c.execute('INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)',(u,'s','2025-01-01','','','BTC’s Rise','',b.digest('BTC’s Rise\n')))
  result={'scope':'btc_general','label':'positive','evidence':"BTC's Rise",'reason':'rise'}
  c.execute('INSERT INTO score_attempts VALUES(1,?,?,?,?,?,?,?)',('', 'old','a',b.digest('BTC’s Rise\n'),json.dumps({'prompt':json.dumps({'text':'BTC’s Rise\n'})}),json.dumps({'done':True,'done_reason':'stop','response':json.dumps(result)}),'mismatch'))
  exact=dict(result,evidence='BTC’s Rise')
  c.execute('INSERT INTO scores VALUES(?,?,?)',('b','old',json.dumps(exact)))
  run,counts=r.revalidate(c,'old')
  self.assertEqual(counts,{'copied':1,'recovered':1,'rejected':0,'unattempted':1})
  self.assertEqual(r.revalidate(c,'old')[0],run)
  self.assertEqual(c.execute('SELECT count(*) FROM scores WHERE run=?',(run,)).fetchone()[0],2)
  self.assertEqual(c.execute('SELECT result FROM scores WHERE run=?',('old',)).fetchone()[0],json.dumps(exact))
  c.close()

if __name__=='__main__':unittest.main()
