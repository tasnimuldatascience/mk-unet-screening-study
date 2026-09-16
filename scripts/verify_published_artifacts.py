"""Validate public evidence without raw data or model checkpoints."""
import csv
import json
import math
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]

def verify_run(name):
    run = ROOT / 'runs' / name
    metrics = json.loads((run / 'metrics.json').read_text())
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    assert metrics['status'] == 'completed'
    assert metrics['epochs'] == 200
    assert [row['epoch'] for row in history] == list(range(1, 201))
    assert metrics['best_epoch'] == max(history, key=lambda row: row['val_dice'])['epoch']
    for key, filename in [('normalized', 'per_image.csv'), ('fixed_threshold', 'per_image_fixed_threshold.csv')]:
        rows = list(csv.DictReader((run / 'test' / filename).open()))
        assert len(rows) == metrics['test'][key]['n']
        for metric in ('dice', 'iou'):
            mean = sum(float(row[metric]) for row in rows) / len(rows)
            assert math.isfinite(mean)
            assert abs(mean - metrics['test'][key][metric]) < 1e-10

if __name__ == '__main__':
    verify_run('clinicdb_reference_seed42')
    verify_run('cwfid_domain_seed42')
    expected = {
        'experimental_execution_summary.pdf': 1,
        'technical_report.pdf': 3,
        'research_proposal.pdf': 6,
    }
    for filename, pages in expected.items():
        assert len(PdfReader(ROOT / 'output' / 'pdf' / filename).pages) == pages
    print('Published evidence and PDF page counts verified.')
