import csv
import hashlib
import json
import math
from pathlib import Path
import fitz

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    for name in ['clinicdb_reference_seed42','cwfid_domain_seed42']:
        run=ROOT/'runs'/name; m=json.loads((run/'metrics.json').read_text())
        assert m['status']=='completed' and m['epochs']==200
        assert m['checkpoint_sha256']==hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
        hist=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
        assert [h['epoch'] for h in hist]==list(range(1,201))
        assert m['best_epoch']==max(hist,key=lambda h:h['val_dice'])['epoch']
        for key,file in [('normalized','per_image.csv'),('fixed_threshold','per_image_fixed_threshold.csv')]:
            records=list(csv.DictReader((run/'test'/file).open()))
            assert len(records)==m['test'][key]['n']
            for metric in ('dice','iou'):
                avg=sum(float(x[metric]) for x in records)/len(records)
                assert math.isfinite(avg) and abs(avg-m['test'][key][metric])<1e-10
        print(name,'evidence verified')
    expected={'technical_report.pdf':3,'research_proposal.pdf':6,'experimental_execution_summary.pdf':1}
    render=ROOT/'tmp/pdfs/final';render.mkdir(parents=True,exist_ok=True)
    for name,count in expected.items():
        doc=fitz.open(ROOT/'output/pdf'/name)
        assert len(doc)==count,(name,len(doc),count)
        for i,page in enumerate(doc):
            assert len(page.get_text())>200
            page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(render/f'{Path(name).stem}-{i+1}.png')
        (ROOT/'output'/f'{Path(name).stem}.txt').write_text('\n\n'.join(p.get_text() for p in doc),encoding='utf-8')
        print(name,len(doc),'pages rendered')
