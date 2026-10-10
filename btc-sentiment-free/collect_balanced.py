#!/usr/bin/env python3
"""Balanced collection companion; reuses the existing scraper, never changes scores."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import sqlite3
import re
import btc_sentiment as core


def bucket(row, start, end):
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', row['url'])
    value = '-'.join(match.groups()) if match else (row['hint'] or '')[:10]
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return 'unknown'
    return value[:7] if start <= value <= end else 'unknown'


def setup(c):
    c.execute('''CREATE TABLE IF NOT EXISTS collection_attempts_v2(
      id INTEGER PRIMARY KEY, at TEXT, start TEXT, end TEXT, url TEXT,
      source TEXT, bucket TEXT, status TEXT, error TEXT)''')
    c.commit()


def schedule(c, start, end, limit, retry=False):
    # Successful collection is global; failures/out-of-period decisions are range specific.
    rows = c.execute('''SELECT * FROM candidates WHERE url NOT IN
                         (SELECT url FROM articles)''').fetchall()
    history = c.execute('''SELECT url,source,bucket,status FROM collection_attempts_v2
                           WHERE start=? AND end=?''', (start,end)).fetchall()
    last = {}; source_load = Counter(); bucket_load = Counter(); attempts = Counter()
    for r in history:
        last[r['url']] = r['status']; attempts[r['url']] += 1
        source_load[r['source']] += 1; bucket_load[r['source'],r['bucket']] += 1
    legacy_errors = {r[0] for r in c.execute("SELECT DISTINCT url FROM errors WHERE stage='article'")}
    queues = defaultdict(list)
    for r in rows:
        status = last.get(r['url'])
        if status == 'outside_period':
            continue
        failed = status == 'failed' or (status is None and r['url'] in legacy_errors)
        if failed and not retry:
            continue
        queues[r['source'], bucket(r,start,end)].append((failed, attempts[r['url']],
            hashlib.sha256(r['url'].encode()).hexdigest(), dict(r)))
    for q in queues.values():
        q.sort(key=lambda x:x[:3], reverse=True)
    result = []
    while queues and len(result) < limit:
        # Every source receives turns, including sources with only unknown dates.
        source = min({s for s,b in queues}, key=lambda s:(source_load[s],s))
        keys = [k for k in queues if k[0] == source]
        # Exhaust fresh work for this source before retrying failures.
        key = min(keys, key=lambda k:(queues[k][-1][0], bucket_load[k],k[1]))
        row = queues[key].pop()[3]
        result.append((row,key[1]))
        source_load[source] += 1; bucket_load[key] += 1
        if not queues[key]: del queues[key]
    return result


def coverage(c, start, end):
    total_days = (dt.date.fromisoformat(end)-dt.date.fromisoformat(start)).days+1
    rows = c.execute('SELECT source,published FROM articles WHERE substr(published,1,10) BETWEEN ? AND ?', (start,end)).fetchall()
    print('COVERAGE publication dates in UTC')
    print(json.dumps({'articles':len(rows),'days_with_articles':len({r['published'][:10] for r in rows}),
                      'total_days':total_days,'missing_days':total_days-len({r['published'][:10] for r in rows})}))
    for source in sorted(set(core.SOURCES) | {r['source'] for r in rows}):
        subset = [r for r in rows if r['source']==source]
        print(json.dumps({'source':source,'candidates':c.execute('SELECT count(*) FROM candidates WHERE source=?',(source,)).fetchone()[0],
          'articles':len(subset),'days':len({r['published'][:10] for r in subset}),
          'by_month':dict(sorted(Counter(r['published'][:7] for r in subset).items())),
          'failed_attempts_v2':c.execute("SELECT count(*) FROM collection_attempts_v2 WHERE source=? AND start=? AND end=? AND status='failed'",(source,start,end)).fetchone()[0]}))
    print('Candidates use URL screening; counts do not establish complete publisher coverage.')


def collect(c, client, plan, start, end):
    counts=Counter()
    for row, month in plan:
        status='failed'; failure=None
        try:
            published,modified,title,summary=core.extract(client.get(row['url']))
            if not start <= published[:10] <= end:
                status='outside_period'
            else:
                text=title+'\n'+summary
                c.execute('INSERT OR IGNORE INTO articles VALUES(?,?,?,?,?,?,?,?)',
                  (row['url'],row['source'],published,modified,core.now(),title,summary,core.digest(text)))
                status='saved'
        except Exception as e:
            failure=str(e)
            print(f"fetch failed {row['source']} {row['url']}: {failure}",flush=True)
        c.execute('INSERT INTO collection_attempts_v2(at,start,end,url,source,bucket,status,error) VALUES(?,?,?,?,?,?,?,?)',
                  (core.now(),start,end,row['url'],row['source'],month,status,failure))
        c.commit(); counts[status]+=1
        print('progress '+json.dumps(dict(counts)),flush=True)
    print('SUMMARY '+json.dumps(dict(counts)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','collect','coverage'])
    p.add_argument('--db',default='btc_sentiment.sqlite')
    p.add_argument('--cache',default='cache')
    p.add_argument('--start',default='2025-01-01'); p.add_argument('--end',default='2025-12-31')
    p.add_argument('--limit',type=int,default=100)
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if dt.date.fromisoformat(a.end)<dt.date.fromisoformat(a.start) or a.limit<=0: p.error('invalid dates or limit')
    # Refuse typo paths instead of silently creating an empty database.
    from pathlib import Path
    if not Path(a.db).is_file(): p.error('database does not exist; use the existing project database')
    c=core.connect(a.db); setup(c)
    if a.command=='coverage': coverage(c,a.start,a.end)
    else:
        plan=schedule(c,a.start,a.end,a.limit,a.retry_failed)
        print('PLAN '+json.dumps({'attempts':len(plan),'sources':dict(Counter(r['source'] for r,b in plan)),
                                 'source_month_hints':dict(Counter(r['source']+'/'+b for r,b in plan))},sort_keys=True))
        print('Months are scheduling hints, not verified publication dates. Unknown dates get a turn.')
        if a.command=='plan':
            for r,b in plan: print(r['source'],b,r['url'])
        else:
            collect(c,core.Client(a.cache),plan,a.start,a.end); coverage(c,a.start,a.end)
    c.close()

if __name__=='__main__': main()
