#!/usr/bin/env python3
"""Offline evidence revalidation. No model calls; source runs remain untouched."""
import argparse
import copy
import json
import re
import sqlite3
import btc_sentiment as b

VERSION = 'quotes_case_whitespace_v1'
QUOTES = str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"'})


def normalized(text):
    out, positions = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            if out and out[-1] == ' ':
                positions[-1] = (positions[-1][0], i+1)
                continue
            folded = ' '
        else:
            folded = ch.translate(QUOTES).casefold()
        for f in folded:
            out.append(f); positions.append((i,i+1))
    return ''.join(out), positions


def match_span(text, evidence):
    if not isinstance(evidence,str) or not evidence.strip():
        raise ValueError('missing evidence')
    if evidence in text:
        start=text.index(evidence)
        return start,start+len(evidence),'exact'
    source,mapping=normalized(text)
    needle,_=normalized(evidence)
    offset=source.find(needle)
    if offset < 0:
        raise ValueError('evidence still differs beyond quotes/case/whitespace')
    start,end=mapping[offset][0],mapping[offset+len(needle)-1][1]
    if normalized(text[start:end])[0] != needle:
        raise ValueError('partial Unicode character match')
    return start,end,VERSION


def repair(result,text):
    if not isinstance(result,dict) or set(result)!=set(b.SCHEMA['required']):
        raise ValueError('invalid output fields')
    start,end,method=match_span(text,result['evidence'])
    fixed=copy.deepcopy(result)
    fixed['evidence']=text[start:end]
    b.validate(fixed,text)
    fixed['evidence_audit']={'model_evidence':result['evidence'],
        'source_evidence':text[start:end],'start':start,'end':end,'method':method}
    return fixed


def revalidate(c,source_run):
    row=c.execute('SELECT metadata FROM runs WHERE id=?',(source_run,)).fetchone()
    if not row:raise ValueError('unknown source run')
    meta={'version':'offline_revalidation_v1','source_run':source_run,
          'source_metadata':json.loads(row[0]),'validator':VERSION}
    run=b.digest(json.dumps(meta,sort_keys=True))
    # Separate deterministic run: rerunning adds new results as source processing advances.
    c.execute('CREATE TABLE IF NOT EXISTS evidence_revalidation(run TEXT,url TEXT,source_attempt INTEGER,status TEXT,detail TEXT,PRIMARY KEY(run,url))')
    c.execute('INSERT OR IGNORE INTO runs VALUES(?,?)',(run,json.dumps(meta)))
    articles=c.execute('SELECT * FROM articles ORDER BY published,url').fetchall()
    counts={'copied':0,'recovered':0,'rejected':0,'unattempted':0}
    for article in articles:
        url=article['url'];text=article['title']+'\n'+article['summary']
        saved=c.execute('SELECT result FROM scores WHERE run=? AND url=?',(source_run,url)).fetchone()
        attempt=None
        try:
            if saved:
                result=json.loads(saved[0]);status='copied'
            else:
                attempt=c.execute('SELECT * FROM score_attempts WHERE run=? AND url=? ORDER BY id DESC LIMIT 1',(source_run,url)).fetchone()
                if not attempt:
                    counts['unattempted']+=1;continue
                request=json.loads(attempt['request'])
                if json.loads(request['prompt'])['text']!=text or attempt['text_hash']!=b.digest(text):
                    raise ValueError('stored input differs from current article')
                response=json.loads(attempt['response'])
                if not isinstance(response,dict) or response.get('done') is not True or response.get('done_reason')=='length':
                    raise ValueError('missing or incomplete generation')
                result=json.loads(response.get('response',''));status='recovered'
            fixed=repair(result,text)
            fixed['evidence_audit']['source_run']=source_run
            fixed['evidence_audit']['source_attempt']=attempt['id'] if attempt else None
            c.execute('INSERT OR REPLACE INTO scores VALUES(?,?,?)',(url,run,json.dumps(fixed)))
            detail=fixed['evidence_audit']['method']
        except (ValueError,KeyError,TypeError) as e:
            status='rejected';detail=str(e)
            c.execute('DELETE FROM scores WHERE url=? AND run=?',(url,run))
        counts[status]+=1
        c.execute('INSERT OR REPLACE INTO evidence_revalidation VALUES(?,?,?,?,?)',
            (run,url,attempt['id'] if attempt else None,status,detail))
    c.commit()
    return run,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',default='btc_sentiment.sqlite');p.add_argument('--source-run',required=True)
    a=p.parse_args()
    # Do not accidentally create a blank DB after a typo.
    from pathlib import Path
    if not Path(a.db).is_file():p.error('database does not exist')
    c=b.connect(a.db)
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'score_attempts' not in tables:p.error('requires scorer v1 database with saved attempts')
    run,counts=revalidate(c,a.source_run)
    print('REVALIDATED_RUN_ID='+run)
    print(json.dumps(counts))
    print('Evidence matching only: sentiment labels have NOT been reviewed for accuracy.')
    c.close()

if __name__=='__main__':main()
