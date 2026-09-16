# Screening requirements implemented

This repository implements the technical work requested by a two-step student
screening brief.

## Step 1: experimental verification

The brief asks applicants to select a listed paper, study its methodology, and
complete two experiments using the corresponding open-source implementation:

1. Replicate a published benchmark experiment.
2. Train and evaluate the architecture on a new neuroimaging or non-biomedical
   benchmark.

The selected paper is MK-UNet. ClinicDB is the reference benchmark and CWFID is
the new precision-agriculture domain. The required execution summary and concise
technical report are published under `output/pdf/`.

## Step 2: research proposal

The brief requests a five-to-six-page proposal aligned with at least one research
pillar and containing motivation, problem definition, related-work gaps, proposed
methods, a framework figure, expected outcomes, and preliminary results.

The six-page proposal addresses continuous adaptation, security and reliability,
plus scalable and explainable AI. Its proposed adaptation method is future work;
only the two backbone experiments are presented as completed evidence.

## Repository scope

The repository contains the complete technical implementation and supporting
evidence. The original screening PDF and licensed datasets are excluded.

## Evidence map

| Screening requirement | Repository evidence |
|---|---|
| Reference replication | ClinicDB 200-epoch run, configuration, history, per-image metrics, and provenance |
| New-domain experiment | CWFID 200-epoch run, configuration, history, per-image metrics, and provenance |
| Brief execution summary | `output/pdf/experimental_execution_summary.pdf` |
| Two-to-three-page technical report | `output/pdf/technical_report.pdf` (3 pages) |
| Core contribution and bottleneck analysis | Technical report, pages 1–2 |
| Cross-domain behavior and replication findings | Technical report, pages 1 and 3 |
| Working code and reproduction instructions | `screening/`, `scripts/`, configurations, and `README.md` |
| Five-to-six-page research proposal | `output/pdf/research_proposal.pdf` (6 pages) |
| Motivation and significance | Proposal, page 1 |
| Problem definition and related-work gap | Proposal, page 2 |
| Proposed method and framework diagram | Proposal, page 3 |
| Evaluation protocol | Proposal, page 4 |
| Expected outcomes and preliminary results | Proposal, page 5 |
| Work plan and references | Proposal, page 6 |
