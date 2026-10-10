import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import full_v3_baseline as baseline


class FullV3BaselineTests(unittest.TestCase):
    def identity(self, articles_hash="articles"):
        return {"version":"pilot_v3","model":baseline.MODEL,"model_digest":"digest",
                "prompt":"frozen","schema":{},"options":baseline.OPTIONS,"think":False,
                "articles_hash":articles_hash,"article_count":2,"max_attempts_per_article":1,
                "run_id":"run"}

    def items(self):
        rows=[]
        for url,title,source in [("a","Same headline","bitcoincom"),("b","Same headline","coindesk")]:
            row={"type":"article","url":url,"source":source,"published":"2025-01-01T12:00:00+00:00",
                 "modified":"2025-01-01T13:00:00+00:00","collected":"2025-02-01T00:00:00+00:00",
                 "title":title,"summary":"summary","text_hash":""}
            row["input_text_hash"]=baseline.text_hash(row);row["text_hash"]=row["input_text_hash"];rows.append(row)
        return rows

    def write_manifest(self,path,items):
        head={"type":"manifest","format":1,"article_count":len(items),"articles_hash":baseline.digest(items),
              "order":"URL ascending","source_database":"fixture","frozen_backup":"fixture"}
        path.write_text("\n".join([baseline.canonical(head),*(baseline.canonical(x) for x in items)])+"\n")
        return head

    def test_resume_identity_and_inputs_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/"results.sqlite"; items=self.items(); identity=self.identity()
            connection=baseline.open_results(db); baseline.assert_state(connection,identity,items)
            baseline.assert_state(connection,identity,items)
            changed=dict(identity);changed["model_digest"]="other"
            with self.assertRaisesRegex(RuntimeError,"identity differs"):
                baseline.assert_state(connection,changed,items)
            altered=[dict(x) for x in items];altered[0]["input_text_hash"]="other"
            with self.assertRaisesRegex(RuntimeError,"inputs differ"):
                baseline.assert_state(connection,identity,altered)
            connection.close()

    def test_aggregation_contributions(self):
        self.assertEqual(baseline.contribution({"scope":"usd_explicit","label":"positive"}),("positive",1,1))
        self.assertEqual(baseline.contribution({"scope":"btc_general","label":"neutral"}),("neutral",0,1))
        self.assertEqual(baseline.contribution({"scope":"relative_only","label":"uncertain"}),("excluded",0,0))
        self.assertEqual(baseline.contribution({"scope":"uncertain","label":"uncertain"}),("uncertain",0,0))
        self.assertEqual(baseline.contribution(None),("failed_unscored",0,0))

    def test_export_deduplicates_and_keeps_missing_score_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);items=self.items();manifest=root/"manifest.jsonl";head=self.write_manifest(manifest,items)
            identity=self.identity(head["articles_hash"]);identity["run_id"]=baseline.digest({k:v for k,v in identity.items() if k!="run_id"})
            db=root/"results.sqlite";connection=baseline.open_results(db);baseline.assert_state(connection,identity,items)
            for item,label in zip(items,("positive","negative")):
                result={"scope":"btc_general","label":label,"horizon":"current_or_past","evidence_ids":["H"],"reason":"x","evidence":[{"id":"H","text":item["title"]}]}
                connection.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?,?)",(item["url"],item["input_text_hash"],"success",baseline.canonical(result),None,None,"fixture"))
            connection.commit();connection.close()
            args=argparse.Namespace(manifest=manifest,results=db,articles=root/"articles.csv",daily=root/"daily.csv",summary=root/"summary.json")
            baseline.export(args)
            with open(args.daily) as handle: rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),365);self.assertEqual(rows[0]["usable_articles"],"1")
            self.assertEqual(rows[0]["sentiment_score"],"100.0")
            self.assertEqual(rows[1]["sentiment_score"],"")
            summary=json.loads(Path(args.summary).read_text())
            self.assertEqual(summary["distribution_after_headline_dedup"]["duplicate_headline"],1)


if __name__ == "__main__": unittest.main()
