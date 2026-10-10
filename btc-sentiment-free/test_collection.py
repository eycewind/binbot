import unittest
from types import SimpleNamespace
import collect_balanced as m

class Tests(unittest.TestCase):
    def setUp(self):
        self.c=m.core.connect(':memory:'); m.setup(self.c)
        for source in ('bitcoincom','coindesk'):
            for month in range(1,13):
                for i in range(4):
                    self.c.execute('INSERT INTO candidates VALUES(?,?,?)',
                      (f'https://{source}/{month}/{i}-bitcoin',source,f'2025-{month:02d}-01'))
        self.c.commit()
    def tearDown(self): self.c.close()
    def test_fair_sources_months_deterministic(self):
        p=m.schedule(self.c,'2025-01-01','2025-12-31',24)
        self.assertEqual(p,m.schedule(self.c,'2025-01-01','2025-12-31',24))
        self.assertEqual(len({(r['source'],b) for r,b in p}),24)
    def test_failure_and_resume(self):
        row,b=m.schedule(self.c,'2025-01-01','2025-12-31',1)[0]
        self.c.execute('INSERT INTO collection_attempts_v2(at,start,end,url,source,bucket,status) VALUES(?,?,?,?,?,?,?)',
          ('now','2025-01-01','2025-12-31',row['url'],row['source'],b,'failed'))
        p=m.schedule(self.c,'2025-01-01','2025-12-31',200)
        self.assertNotIn(row['url'],[r['url'] for r,b in p])
        self.assertNotEqual(p[0][0]['source'],row['source'])
        p=m.schedule(self.c,'2025-01-01','2025-12-31',200,True)
        self.assertIn(row['url'],[r['url'] for r,b in p])
    def test_actual_date_and_existing_scores_preserved(self):
        p=m.schedule(self.c,'2025-01-01','2025-12-31',1)
        raw=b'<script type="application/ld+json">{"@type":"NewsArticle","headline":"Bitcoin test","datePublished":"2024-12-31T23:00:00Z"}</script>'
        self.c.execute('INSERT INTO scores VALUES(?,?,?)',('existing','run','{}'))
        m.collect(self.c,SimpleNamespace(get=lambda url:raw),p,'2025-01-01','2025-12-31')
        self.assertEqual(self.c.execute('SELECT count(*) FROM articles').fetchone()[0],0)
        self.assertEqual(self.c.execute('SELECT count(*) FROM scores').fetchone()[0],1)
        self.assertNotIn(p[0][0]['url'],[r['url'] for r,b in m.schedule(self.c,'2025-01-01','2025-12-31',200,True)])
    def test_hint_not_filter_and_legacy_errors(self):
        self.c.execute('INSERT INTO candidates VALUES(?,?,?)',('https://x/bitcoin','third','2026-01-01'))
        p=m.schedule(self.c,'2025-01-01','2025-12-31',200)
        self.assertIn(('https://x/bitcoin','unknown'),[(r['url'],b) for r,b in p])
        self.c.execute('INSERT INTO errors VALUES(?,?,?,?)',('now','article','https://x/bitcoin','error'))
        self.assertNotIn('https://x/bitcoin',[r['url'] for r,b in m.schedule(self.c,'2025-01-01','2025-12-31',200)])

if __name__=='__main__': unittest.main()
