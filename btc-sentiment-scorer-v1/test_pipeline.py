import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from btc_sentiment import allowed, connect, daily, extract, validate


class PipelineTests(unittest.TestCase):
    def test_robots_wildcards_and_specific_agent(self):
        rules = 'User-agent: *\nDisallow: /api/*\nAllow: /api/public$\n'
        self.assertFalse(allowed(rules, 'https://example.com/api/private'))
        self.assertTrue(allowed(rules, 'https://example.com/api/public'))
        self.assertFalse(allowed(rules, 'https://example.com/api/public/more'))
        rules += '\nUser-agent: BTCResearch\nDisallow: /\n'
        self.assertFalse(allowed(rules, 'https://example.com/article'))

    def test_metadata_utc_and_publication_not_modified(self):
        doc = {'@type': 'NewsArticle', 'headline': 'Bitcoin rises', 'description': 'BTC reaches $100,000.',
               'datePublished': '2026-01-01T00:30:00+05:00', 'dateModified': '2026-05-01T00:00:00Z'}
        raw = ('<script type="application/ld+json">' + json.dumps(doc) + '</script>').encode()
        published, modified, title, _ = extract(raw)
        self.assertEqual(published[:10], '2025-12-31')
        self.assertTrue(modified.startswith('2026-05'))
        self.assertEqual(title, 'Bitcoin rises')

    def test_reject_hallucinated_evidence(self):
        with self.assertRaises(ValueError):
            validate({'scope': 'usd_explicit', 'label': 'positive', 'evidence': 'invented', 'reason': 'x'}, 'BTC falls')

    def test_daily_excludes_cross_pair_keeps_missing_and_dedups(self):
        c = connect(':memory:')
        c.execute('INSERT INTO runs VALUES(?,?)', ('r', '{}'))
        items = [('a','BTC rises','usd_explicit','positive'), ('b','BTC rises','usd_explicit','positive'),
                 ('c','ETH beats BTC','relative_only','negative'), ('d','BTC falls','usd_explicit','negative')]
        for u,t,scope,label in items:
            c.execute('INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)',
                      (u,'source','2025-01-01T12:00:00+00:00','','',t,'',u))
            c.execute('INSERT INTO scores VALUES(?,?,?)', (u,'r',json.dumps({'scope':scope,'label':label})))
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'daily.csv'
            daily(c, argparse.Namespace(run='r',start='2025-01-01',end='2025-12-31',output=f))
            with f.open() as handle: rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows),365)
        self.assertEqual(float(rows[0]['sentiment_score']),0)
        self.assertEqual(rows[0]['positive'],'1')
        self.assertEqual(rows[0]['negative'],'1')
        self.assertEqual(rows[0]['excluded'],'1')
        self.assertEqual(rows[1]['sentiment_score'],'')
        self.assertEqual(rows[1]['status'],'no_collected_articles')
        c.close()


if __name__ == '__main__': unittest.main()
