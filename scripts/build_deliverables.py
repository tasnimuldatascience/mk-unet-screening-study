"""Build evidence-linked PDFs and editable text from completed runs only."""
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=10,textColor=colors.HexColor('#223047')))
styles.add(ParagraphStyle(name='Kicker',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=colors.HexColor('#087f8c'),spaceAfter=10))
styles.add(ParagraphStyle(name='Reference',fontName='Helvetica',fontSize=7.6,leading=9.2,spaceAfter=3,textColor=colors.HexColor('#223047')))
styles.add(ParagraphStyle(name='Byline',fontName='Helvetica',fontSize=9.5,leading=12,spaceAfter=10,textColor=colors.HexColor('#52647a')))
styles['Title'].fontName='Helvetica-Bold'; styles['Title'].fontSize=20; styles['Title'].leading=24; styles['Title'].textColor=colors.HexColor('#142d4e')
styles['Heading1'].fontSize=18; styles['Heading1'].leading=22; styles['Heading1'].textColor=colors.HexColor('#142d4e')
styles['Heading2'].fontSize=12; styles['Heading2'].leading=16; styles['Heading2'].textColor=colors.HexColor('#087f8c')

def p(text): return Paragraph(text,styles['BodyCustom'])
def refp(text): return Paragraph(text,styles['Reference'])
def h(text): return Paragraph(text,styles['Heading2'])
def title(kicker,text): return [Paragraph(kicker.upper(),styles['Kicker']),Paragraph(text,styles['Title']),Spacer(1,12)]
def table(rows,widths=None):
    t=Table([[p(str(x)) for x in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf2f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),3),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#87a9ba'))]))
    return t

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#bdd0db')); canvas.line(48,39,564,39)
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#52647a'))
    canvas.drawString(48,27,'MK-UNet | Reproducible segmentation study')
    canvas.drawRightString(564,27,str(doc.page))

def build(name,pages):
    story=[]
    for i,page in enumerate(pages):
        if i: story.append(PageBreak())
        story.extend(page)
    document_titles = {
        'experimental_execution_summary.pdf': 'MK-UNet Experimental Execution Summary',
        'technical_report.pdf': 'MK-UNet Technical Report',
        'research_proposal.pdf': 'Risk-Budgeted Reversible Adaptation Research Proposal',
    }
    SimpleDocTemplate(str(OUT/name),pagesize=(612,792),rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=53,
                      title=document_titles[name],author='Tasnimul Hasan').build(story,onFirstPage=footer,onLaterPages=footer)

def load(name):
    path=ROOT/'runs'/name/'metrics.json'
    if not path.exists(): raise RuntimeError(f'Complete training before generating results: {path}')
    return json.loads(path.read_text())

def diagram():
    d=Drawing(510,178)
    navy, blue, teal, amber, green, red, ink = [colors.HexColor(x) for x in
        ('#142d4e','#2563a6','#087f8c','#d97706','#15803d','#c62828','#24364b')]

    def arrow(x1,y1,x2,y2,color=ink):
        d.add(Line(x1,y1,x2,y2,strokeColor=color,strokeWidth=1.6))
        if abs(x2-x1) >= abs(y2-y1):
            s = 1 if x2 > x1 else -1
            d.add(Polygon([x2,y2,x2-6*s,y2+3.5,x2-6*s,y2-3.5],fillColor=color,strokeColor=color))
        else:
            s = 1 if y2 > y1 else -1
            d.add(Polygon([x2,y2,x2-3.5,y2-6*s,x2+3.5,y2-6*s],fillColor=color,strokeColor=color))

    def card(x,y,w,h,fill,stroke,heading,sub,dark=True):
        d.add(Rect(x,y,w,h,rx=7,fillColor=fill,strokeColor=stroke,strokeWidth=1.2))
        text_color = ink if dark else colors.white
        d.add(String(x+w/2,y+h-15,heading,textAnchor='middle',fontName='Helvetica-Bold',fontSize=8.2,fillColor=text_color))
        d.add(String(x+w/2,y+9,sub,textAnchor='middle',fontName='Helvetica',fontSize=6.8,fillColor=text_color))

    d.add(Rect(137,157,236,18,rx=9,fillColor=colors.HexColor('#e6f6f7'),strokeColor=teal))
    d.add(String(255,163,'CALIBRATED ON SOURCE VALIDATION SHIFTS',textAnchor='middle',fontName='Helvetica-Bold',fontSize=7.2,fillColor=teal))

    card(0,91,78,42,navy,navy,'SHIFTED IMAGE','unlabeled x',dark=False)
    card(93,91,78,42,colors.HexColor('#e8f1fb'),blue,'FROZEN','baseline p0')
    card(186,91,78,42,colors.HexColor('#fff3df'),amber,'SHADOW','candidate p1')
    card(279,91,86,42,colors.HexColor('#e4f5f5'),teal,'BENEFIT LCB','compare p1 to p0')

    d.add(Polygon([384,112,424,142,464,112,424,82],fillColor=ink,strokeColor=ink))
    d.add(String(424,117,'POSITIVE?',textAnchor='middle',fontName='Helvetica-Bold',fontSize=7.4,fillColor=colors.white))
    d.add(String(424,105,'within risk budget?',textAnchor='middle',fontSize=6.2,fillColor=colors.white))

    card(296,12,94,35,colors.HexColor('#fdeaea'),red,'ROLL BACK','use frozen p0')
    card(416,12,89,35,colors.HexColor('#e8f6ec'),green,'COMMIT','use candidate p1')

    arrow(78,112,93,112,blue)
    arrow(171,112,186,112,amber)
    arrow(264,112,279,112,teal)
    arrow(365,112,384,112,ink)
    d.add(Line(424,82,424,66,strokeColor=ink,strokeWidth=1.6))
    d.add(Line(343,66,461,66,strokeColor=ink,strokeWidth=1.6))
    arrow(343,66,343,47,red)
    arrow(461,66,461,47,green)
    d.add(String(334,71,'NO',fontName='Helvetica-Bold',fontSize=6.5,fillColor=red))
    d.add(String(466,71,'YES',fontName='Helvetica-Bold',fontSize=6.5,fillColor=green))
    d.add(String(255,1,'A rejected candidate leaves the deployed model unchanged.',textAnchor='middle',fontSize=7.4,fillColor=colors.HexColor('#52647a')))
    return d

def main():
    clinic=load('clinicdb_reference_seed42'); crop=load('cwfid_domain_seed42')
    results={'ClinicDB':clinic,'CWFID':crop}
    rows=[['Experiment','Test n','Dice / IoU','Best epoch']]
    for name,r in results.items():
        m=r['test']['normalized']; rows.append([name,m['n'],f"{m['dice']*100:.2f}% / {m['iou']*100:.2f}%",r['best_epoch']])
    result_table=lambda:table(rows,[120,65,210,121])
    delta=100*clinic['test']['normalized']['dice']-93.48
    quantitative=f"The ClinicDB run achieved {clinic['test']['normalized']['dice']*100:.2f}% Dice and {clinic['test']['normalized']['iou']*100:.2f}% IoU. Relative to the published 93.48% Dice reference [1], the difference is {delta:+.2f} percentage points. This comparison is descriptive: the publication averages five runs, whereas this study uses one seed and corrects evaluation and resizing behavior in the current public code."
    crop_text=f"On CWFID, the independently trained model achieved {crop['test']['normalized']['dice']*100:.2f}% Dice and {crop['test']['normalized']['iou']*100:.2f}% IoU on 21 held-out images. The target is all vegetation, merging crops and weeds. These results demonstrate that the architecture can be trained on an agricultural task; they do not establish zero-shot medical-to-agricultural transfer or robustness across farms."
    fixed=f"With a fixed sigmoid threshold of 0.5, test Dice is {clinic['test']['fixed_threshold']['dice']*100:.2f}% on ClinicDB and {crop['test']['fixed_threshold']['dice']*100:.2f}% on CWFID. The primary scores instead use the upstream per-image min-max normalization followed by 0.5 thresholding. This extra result exposes sensitivity to output scaling."
    refs=[
      '[1] Rahman and Marculescu. MK-UNet. ICCV Workshops, 2025. <link href="https://arxiv.org/abs/2509.18493" color="#087f8c">Paper</link>; <link href="https://github.com/SLDGroup/MK-UNet" color="#087f8c">official code</link>.',
      '[2] Haug and Ostermann. A Crop/Weed Field Image Dataset. ECCV Workshops, 2015. <link href="https://doi.org/10.1007/978-3-319-16220-1_8" color="#087f8c">Dataset paper</link>; <link href="https://github.com/cwfid/dataset" color="#087f8c">data and split</link>.',
      '[3] Wang et al. Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR, 2021. <link href="https://arxiv.org/abs/2006.10726" color="#087f8c">Paper</link>.',
      '[4] Niu et al. Efficient Test-Time Model Adaptation without Forgetting. ICML, 2022. <link href="https://proceedings.mlr.press/v162/niu22a.html" color="#087f8c">Paper</link>.',
      '[5] Niu et al. Towards Stable Test-Time Adaptation in Dynamic Wild World. ICLR, 2023. <link href="https://arxiv.org/abs/2302.12400" color="#087f8c">Paper</link>.',
      '[6] Cao et al. GraTa: Gradual Test-Time Adaptation via Gradient Alignment. AAAI, 2025. <link href="https://arxiv.org/abs/2412.03794" color="#087f8c">Paper</link>.',
      '[7] Liu et al. A3-TTA: Adaptive Anchors and Anti-forgetting Test-Time Adaptation. IEEE TIP, 2025. <link href="https://doi.org/10.1109/TIP.2025.3563821" color="#087f8c">Paper</link>.',
      '[8] TEGDA: Test-Time Entropy and Gradient Distribution Alignment for Medical Image Segmentation. MICCAI, 2025. <link href="https://papers.miccai.org/miccai-2025/0281-Paper2180.html" color="#087f8c">Paper</link>.',
      '[9] Bernal et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy. CMIG, 2015. <link href="https://doi.org/10.1016/j.compmedimag.2015.02.007" color="#087f8c">ClinicDB</link>.']
    code='<link href="https://github.com/tasnimuldatascience/mk-unet-screening-study" color="#087f8c">Code repository</link> | <link href="https://github.com/tasnimuldatascience/mk-unet-screening-study/blob/main/README.md" color="#087f8c">Reproduction instructions</link>.'
    report=[
      title('Deliverable 2 / 1 of 3','Lightweight segmentation across medicine and agriculture')+[
        Paragraph('Tasnimul Hasan',styles['Byline']),
        p('This report covers two completed MK-UNet training experiments and the data checks needed to interpret them. Every reported score comes from a saved checkpoint and a per-image evaluation record.'),
        h('Architecture and motivation'),
        p('MK-UNet combines depth-wise convolutions at multiple kernel sizes with a U-shaped encoder and decoder. The model uses channel, spatial, and gated skip attention to refine the representation [1]. Our imported standard model contains 315,566 parameters. The main engineering attraction is a compact segmentation backbone that can be trained without a large pretrained encoder.'),
        h('Experiment design'),
        p('Experiment 1 uses the authors\' ClinicDB split and the reference configuration. Experiment 2 trains a new model from scratch on CWFID [2]. Both use seed 42 and 200 epochs; the checkpoint is selected by validation Dice before test evaluation. The task and dataset change in experiment 2, while the architecture stays fixed.'),
        result_table(),Spacer(1,12),p(quantitative),p(code)],
      title('Deliverable 2 / 2 of 3','What the implementation audit changed')+[
        h('Data integrity precedes metric comparison'),
        p('The ClinicDB download preserves the authors\' split and records file identifiers and SHA-256 hashes. CWFID\'s published split places image 028 in both training and test. We keep it in test and remove it from training. Eight remaining training images become validation samples, yielding 31 training, 8 validation, and 21 test images. Every pair is checked for matching dimensions; duplicate image hashes cannot cross splits.'),
        p('Visual inspection showed that CWFID\'s binary masks use black for vegetation, so the loader explicitly inverts them. A regression test checks the vegetation fraction and prevents soil pixels from being mistaken for the segmentation target.'),
        h('Evaluation and training corrections'),
        p('The upstream loader returns PIL dimensions as width then height, while the evaluator reads them in the opposite order. Our evaluator uses native mask dimensions and preserves original ground truth. The upstream scale loop also overwrites its input between passes; our scale loop always starts from the original batch. These fixes improve protocol clarity but mean the execution is not byte-for-byte identical to the public runner.'),
        p('The paper and repository defaults disagree. The reference configuration follows the paper\'s stated settings: 352-pixel input, AdamW learning rate and weight decay of 0.0001, batch 16, gradient clipping at 0.5, no augmentation, and three training scales. The agricultural run uses batch 8. ClinicDB uses float16 autocast with gradient scaling to fit the GPU; CWFID uses FP32. This precision difference and current CUDA PyTorch are recorded deviations.'),
        p(fixed),h('Bottleneck'),p('Small parameter count does not guarantee proportionally low latency. Multiple depth-wise branches, attention operations, resizing, and GPU launch overhead remain. Model size and this study\'s wall time are observed quantities; mobile-device throughput and energy consumption were not measured.')],
      title('Deliverable 2 / 3 of 3','Findings, limitations, and next experiments')+[
        p(crop_text),
        h('What these experiments establish'),
        p('The code can acquire and validate both datasets, train the original model, select a checkpoint, evaluate native-resolution masks, export predictions, and regenerate the submission documents. Saved manifests, logs, and checkpoint hashes tie each score to an actual run. They also make protocol changes visible instead of folding them into an unexplained performance difference.'),
        h('What remains uncertain'),
        p('A single seed cannot recover training variability. ClinicDB frames may be correlated within acquisition sequences, and patient-level separation is not verified. CWFID is small and comes from one field; its image-level split cannot establish unseen-farm reliability. Bootstrap intervals describe image resampling within the fixed test set, not uncertainty across new hospitals or farms.'),
        h('Next study'),
        p('The proposal develops spatially selective, bounded test-time adaptation. A useful next evaluation would keep source-trained weights fixed as a control, compare entropy adaptation against reliability-gated updates, and test whether rollback prevents degradation under sensor shifts. Hyperparameters must be selected on validation shifts; target test masks must never enter the adaptation objective.'),
        h('Evidence and references'),p('Execution detail: <link href="experimental_execution_summary.pdf" color="#087f8c">experimental execution summary</link>. Run evidence: <link href="../../runs/clinicdb_reference_seed42/metrics.json" color="#087f8c">ClinicDB metrics</link> and <link href="../../runs/cwfid_domain_seed42/metrics.json" color="#087f8c">CWFID metrics</link>.'),p(refs[0]),p(refs[1]),
        p('The public repository contains the configurations, manifests, per-image results, implementation, and reproduction instructions used in this study.')]
    ]
    build('technical_report.pdf',report)
    proposal=[
      title('Research proposal / 1 of 6','Risk-budgeted reversible adaptation for compact segmentation')+[
        Paragraph('Tasnimul Hasan',styles['Byline']),
        p('<b>Research pillars:</b> Continuous Adaptation, Security &amp; Reliability; Scalable, Explainable &amp; Interactive AI.'),
        h('Motivation and significance'),
        p('A segmentation model deployed on a constrained device must cope with changing illumination, acquisition settings, and image quality. A field robot may encounter shadows or blur; a medical imaging system may encounter a different camera response. A compact network can reduce storage and computation, yet those savings do not establish that its predictions remain trustworthy when the input distribution changes.'),
        p('I propose a commit controller that estimates whether an update will help before it changes the deployed model. The update first runs on a shadow copy. Boundary and multi-view consistency measurements are then used to calculate a validation-calibrated lower bound on the expected Dice change. The candidate is committed only when that bound is positive and the clean-retention and latency budgets still have room. Every other candidate is discarded, leaving the frozen model unchanged.'),
        h('Central question and hypothesis'),
        p('The main question is whether a calibrated lower bound on update benefit can reject harmful test-time changes without blocking the useful ones. I expect decisions based on predicted Dice change to be safer than decisions based only on confidence or entropy, while still meeting fixed accuracy-retention and latency limits.'),
        h('Scope and practical value'),
        p('The initial study concerns binary segmentation with MK-UNet and accessible RGB datasets. Medical and agricultural experiments are separate tasks with separate trained models. No claim is made that the semantics of a polyp and a plant are interchangeable. The common object of study is the behavior of the adaptation mechanism, including its failure modes and computational cost.'),
        p('The practical result will be a small, auditable prototype that can run on the available laptop GPU. Clinical use and autonomous crop treatment remain outside this study because each would require independent validation in its operating environment.')],
      title('Research proposal / 2 of 6','Problem definition and research gap')+[
        h('Formal setting'),
        p('Let f(x; theta, phi) output a foreground probability at each pixel. Source training learns backbone weights theta and a small adaptable parameter subset phi. At deployment, an unlabeled target image x arrives from a shifted distribution. The objective is to improve segmentation risk on that target distribution while respecting a latency budget and retaining source-domain behavior. Target labels are reserved for offline assessment.'),
        h('Related work'),
        p('MK-UNet [1] supplies the compact backbone. Tent [3], EATA [4], and SAR [5] establish entropy adaptation, sample selection, and stability controls. GraTa [6] aligns update gradients; A3-TTA [7] uses anchors and anti-forgetting; TEGDA [8] estimates test-time quality for medical segmentation. These methods make a simple uncertainty gate insufficient as a novelty claim.'),
        p('The method differs in what it asks before an update is accepted: will this candidate outperform the frozen prediction? It estimates that difference directly, places a calibrated lower bound around it, and connects the decision to a cumulative risk budget and exact rollback. Calibration uses only source-validation images under shifts fixed in advance. Experiments must still show whether this combination provides a real advantage.'),
        h('Failure cases to make measurable'),
        p('Three cases will be tracked explicitly: nearly empty predictions that minimize entropy without recovering the object; localized boundary errors that are hidden by a good mean image score; and drift after a long run of similar or corrupted frames. A fourth practical failure is exceeding the inference budget because adaptation requires repeated forward and backward passes.'),
        h('Research objectives'),
        p('The work has three objectives: measure how the frozen backbone fails under controlled shifts; test whether calibrated decisions reject harmful updates; and find the lowest update frequency that preserves any adaptation gain. Each experiment includes the same frozen-model control, a matching ablation, and a failure rule chosen before the final test run.')],
      title('Research proposal / 3 of 6','Calibrated commit controller')+[
        diagram(),
        h('Candidate update and benefit features'),
        p('For each image, retain frozen prediction p0 and take one entropy-consistency step on a shadow copy of normalization affine parameters. Compute candidate prediction p1. Features summarize changes in cross-view disagreement, boundary stability, foreground area, entropy, and update norm. A lightweight regressor predicts ΔDice = Dice(p1,y) - Dice(p0,y); target labels are never inputs.'),
        h('Validation calibration'),
        p('Fit the regressor on one portion of source-validation corruptions and use a disjoint calibration portion to estimate a one-sided residual quantile. The commit score is the predicted benefit minus this margin. Commit only when this lower confidence bound is positive, latency is within budget, and the cumulative clean-retention budget is unspent. Otherwise restore the exact frozen snapshot and return p0.'),
        h('Guarantee boundary and audit trail'),
        p('Split-conformal coverage relies on exchangeability between calibration and deployment shifts. It is not guaranteed under an unseen hospital or field distribution. The study therefore reports calibration coverage separately on synthetic validation shifts and on sealed real external data. Every candidate logs its features, bound, decision, latency, parameter hash, and rollback result.'),
        p('A tested prototype in the repository implements the conformal lower-bound calibrator, risk-budget decision, and rollback semantics. Training integration and empirical evaluation remain planned research, keeping completed evidence separate from proposed claims.')],
      title('Research proposal / 4 of 6','Experimental design and evaluation')+[
        h('Data and separation'),
        p('All required data are available locally and reproducible from public sources. ClinicDB provides 489/61/62 source train/validation/test images. CVC-ColonDB provides 380 images sealed for external medical testing; its labels cannot tune the model, calibrator, thresholds, or stopping rules. CWFID provides a separate 31/8/21 agricultural feasibility branch after correcting published train/test overlap. Manifests record every file hash and confirm zero exact-image overlap between ClinicDB and ColonDB.'),
        h('Controls and ablations'),
        p('Compare the frozen model, normalization-affine entropy adaptation, augmentation averaging, and the proposed controller. Ablate the benefit model, conformal margin, cumulative budget, and rollback. Use identical checkpoints, image order, preprocessing, and thresholds. A random-commit control tests whether selection adds value beyond updating at the same frequency.'),
        h('Shift suite'),
        p('Start with preregistered brightness reduction, blur, and additive noise, evaluated independently and in sequences that alternate clean and shifted inputs. Run both episodic adaptation, which resets after each image, and continual adaptation, which retains accepted parameters. Corruption severity is controlled and recorded; synthetic corruption is a diagnostic proxy, not a substitute for real scanner or field shifts.'),
        h('Outcomes and statistics'),
        p('Primary outcomes are mean per-image Dice and the paired Dice change relative to the frozen model under each shift. Report IoU, boundary-sensitive errors, update acceptance rate, clean-domain retention, peak GPU memory, and median and 95th-percentile end-to-end latency. Use at least five training seeds and report seed variability separately from image bootstrap uncertainty. Where grouping metadata exist, resample acquisition groups rather than individual frames.'),
        h('Decision rule'),
        p('A planned feasibility target is a positive paired improvement on shifted data with no more than a one-percentage-point mean Dice loss on clean data. This is a research target, not a measured result. Report every prespecified condition, including failures, and separate method selection on validation data from the final test comparison.')],
      title('Research proposal / 5 of 6','Preliminary evidence and expected outcomes')+[
        h('Completed backbone experiments'),result_table(),Spacer(1,10),p(quantitative),p(crop_text),
        h('How the evidence supports feasibility'),
        p('The preliminary runs verify that this laptop can train and evaluate the selected architecture on both tasks and that the data pipeline can preserve traceable split and checkpoint identities. They support the feasibility of a modest adaptation study. They do not test the proposed gate, demonstrate a causal benefit from adaptation, or establish that the research hypothesis is true.'),
        h('Expected scientific outputs'),
        p('The study will produce an adaptation benchmark, working rollback code, an accuracy-versus-cost comparison, and spatial examples of updates that helped or failed. If the frozen model or augmentation averaging performs better within the compute budget, that result will be reported directly. Settings are frozen before the sealed test evaluation, so the test set cannot become another tuning set.'),
        h('Limits and mitigation'),
        p('The largest risks are calibration-set size, non-exchangeable real shifts, correlated frames, and threshold sensitivity. Report bound coverage, selective risk versus commit rate, and failures on sealed ColonDB in addition to mean accuracy. Multiple seeds and a frozen protocol limit researcher degrees of freedom. Raw probabilities are required for calibration; replication-only min-max normalization is unsuitable for reliability estimation.')],
      [Spacer(1,30)]+title('Research proposal / 6 of 6','Work plan and reproducibility')+[
        table([['Period','Milestone and decision'],['Week 1','Repeat baseline seeds; integrate the shadow update; lock shifts and data partitions.'],['Week 2','Fit the benefit model and calibrator; freeze settings; run ablations and latency tests.'],['Week 3','Evaluate sealed ColonDB and CWFID branches; stress test streams; analyze and write.']],[92,424]),Spacer(1,12),
        h('Reproducibility and resources'),
        p('The project retains upstream provenance, data manifests and hashes, training histories, checkpoint identities, and per-image metrics. ClinicDB, ColonDB, and CWFID are acquired; raw files remain outside Git under their original terms. The adaptation study will add calibration folds, update decisions, bounds, rollback checks, and timing records. The one-step shadow design is implementable on the available RTX 5070 laptop.'),
        p('The proposal describes planned research. Only the baseline results on page 5 are completed experiments; all adaptation results remain future work.'),
        h('References')]+[refp(r) for r in refs]
    ]
    build('research_proposal.pdf',proposal)
    summary=[title('Deliverable 1','Experimental execution summary')+[
        Paragraph('Tasnimul Hasan',styles['Byline']),
        p('Completed single-seed, 200-epoch training runs. Model: standard MK-UNet imported from the pinned official code. Hardware: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB VRAM. Runtime: Python 3.13, PyTorch 2.11.0+cu128.'),result_table(),Spacer(1,12),
        p(f"Reference run: {clinic['elapsed_seconds']/60:.1f} minutes. Agricultural run: {crop['elapsed_seconds']/60:.1f} minutes. Times include training, validation, checkpoint writes, and final evaluation; they are not pure inference timings."),
        p(quantitative),p(fixed),
        h('Setup and execution'),
        p('Run scripts/download_clinicdb.py, then scripts/prepare_data.py all. Train with python -m screening.run train --config configs/clinicdb_reference.json and the corresponding cwfid_domain.json configuration. Use --resume to continue the same configuration from last.pt. See README.md for the exact environment and full workflow.'),
        h('Evidence index'),
        p('Each run directory contains provenance.json, history.jsonl, best.pt, last.pt, metrics.json, test/per_image.csv, and test/predictions/. Dataset manifests record split assignments and file hashes. sources/protocol_audit.md documents deviations, dataset corrections, and interpretation limits. Tests cover metric arithmetic, gradient direction, mask foreground, split overlap, and native-resolution evaluation.'),p(code),
        h('Reproducibility status'),p('The experimental summary, three-page technical report, and six-page proposal are linked to the published configurations, manifests, run histories, per-image measurements, and verification workflow. The repository link appears above.')]]
    build('experimental_execution_summary.pdf',summary)
    (ROOT/'output/results_summary.json').write_text(json.dumps(results,indent=2))
    print('Created three PDFs from completed run evidence.')

if __name__=='__main__':main()
