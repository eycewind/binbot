import argparse,json,unittest
from unittest.mock import patch
import btc_sentiment as b

class ScoringTests(unittest.TestCase):
 def test_response_validation(self):
  for r in [{'done':True,'response':'','thinking':'abc'}, {'done':True,'done_reason':'length','response':'{}'}, {'done':False,'response':'{}'}]:
   with self.assertRaises(ValueError): b.parse_response(r,'BTC rises')
  good={'scope':'usd_explicit','label':'positive','evidence':'BTC rises','reason':'Rising USD price'}
  self.assertEqual(b.parse_response({'done':True,'response':json.dumps(good)},'BTC rises'),good)
 def test_persistence_stop_resume_and_identity(self):
  c=b.connect(':memory:')
  c.execute('INSERT INTO runs VALUES(?,?)',('old','{}'))
  for i in range(5):
   c.execute('INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)',(str(i),'s','2025-01-01','','','BTC rises','',str(i)))
  args=argparse.Namespace(model='test:latest',think='off',num_ctx=4096,num_predict=1024,limit=5,stop_after=3)
  calls=[]
  def api(path,payload=None):
   if path=='/api/tags':return {'models':[{'name':'test:latest','digest':'abc'}]}
   if path=='/api/show':return {'capabilities':['thinking']}
   calls.append(payload)
   return {'done':True,'response':'','thinking':'reasoning','done_reason':'length'}
  with patch.object(b,'local_api',side_effect=api):b.score(c,args)
  self.assertEqual(len(calls),3)
  self.assertIs(calls[0]['think'],False)
  self.assertIsInstance(calls[0]['format'],dict)
  self.assertEqual(c.execute('SELECT count(*) FROM scores').fetchone()[0],0)
  self.assertEqual(c.execute('SELECT count(*) FROM score_attempts').fetchone()[0],3)
  self.assertIn('reasoning',c.execute('SELECT response FROM score_attempts LIMIT 1').fetchone()[0])
  def success(path,payload=None):
   if path!='/api/generate':return api(path,payload)
   return {'done':True,'done_reason':'stop','response':json.dumps({'scope':'btc_general','label':'positive','evidence':'BTC rises','reason':'text rises'})}
  args.limit=1
  with patch.object(b,'local_api',side_effect=success):b.score(c,args)
  self.assertEqual(c.execute('SELECT url FROM scores').fetchone()[0],'3')
  self.assertEqual(c.execute('SELECT count(*) FROM runs').fetchone()[0],2)
  args.num_predict=2048
  with patch.object(b,'local_api',side_effect=success):b.score(c,args)
  self.assertEqual(c.execute('SELECT count(*) FROM runs').fetchone()[0],3)
  c.close()

if __name__=='__main__':unittest.main()
