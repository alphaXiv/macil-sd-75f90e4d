# Visual Contrastive Self-Distillation — bounded reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/macil-sd-75f90e4d/blob/main/notebooks/vcsd_reproduction.py)

Independent reproduction of [Visual Contrastive Self-Distillation (arXiv:2607.21556)](https://arxiv.org/abs/2607.21556) on the public `Qwen/Qwen3-VL-2B-Instruct` checkpoint.

**Verdict: not reproduced in this bounded setup.** The paper reports VCSD above matched OPSD by 3.00 points on MMStar and 1.40 on MathVista. On our fixed public 64-item subsets after 36 updates, VCSD averaged **27.73%** across the two tasks versus **30.47%** for OPSD—a **−2.73-point** gap—and versus **28.91%** for the unchanged checkpoint.

The mechanism was visible despite the accuracy divergence: original images assigned 11.98% mean probability to ground-truth answer-token variants, black images 1.46%, and contrast shaping 29.84%. Removing the plausible-support restriction degraded the 36-step mean by 2.15 points, compared with 1.17 for supported VCSD.

| 36-step evidence | Base | OPSD | VCSD | No contrast | No support |
|---|---:|---:|---:|---:|---:|
| Two-benchmark mean accuracy | 28.91 | 30.47 | 27.73 | 28.13 | 26.76 |
| Change from base | 0.00 | +1.56 | −1.17 | −0.78 | −2.15 |

Scope was deliberately bounded: 48 fixed ViRL39K items rather than 38,870; 64 fixed examples each from MMStar and MathVista rather than seven full benchmarks; batch one and one rollout rather than batch 32 and eight rollouts; 12 or 36 updates rather than the paper’s 90; and 24 rollout / 32 evaluation tokens. Prompts and strict scoring are explicit substitutions because the paper’s VCSD implementation was not public.

- [Illustrated scientific report](reports/vcsd-reproduction/report.md)
- [Self-contained marimo tutorial](notebooks/vcsd_reproduction.py)
- [Measured result data](reports/vcsd-reproduction/data/summary.json)

All evidence ran with the explicit **Kubernetes** backend on **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPUs. Peak concurrency was **16 GPUs**; actual experiment-campaign wall time was **50m10s (0.84 hours)**.

## Experiment log

Every formal node used the exact command shown by `orx exp status`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, and validated implementation | Not run as an experiment (publication surface) | Presentation-only | None |
| [Base final-call repair](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/base-final-call-repair) | Unchanged checkpoint on fixed public subsets | `bash run.sh` | 31.25 MMStar; 26.56 MathVista | Kubernetes, 1 GPU, 2m48s |
| [Matched OPSD seed 0](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/opsd-token-type-repair-seed0) | 12-update answer-hint teacher control; five-seed family | `bash run.sh` | Five-seed mean change +0.47 points | Kubernetes, 1 GPU/run |
| [Supported VCSD seed 0](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/vcsd-token-type-repair-seed0) | 12-update visual contrast with β=.1; five-seed family | `bash run.sh` | Five-seed mean change −0.47 points | Kubernetes, 1 GPU/run |
| [Unrestricted VCSD seed 0](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/beta0-token-type-repair-seed0) | β=0 support ablation; five-seed family | `bash run.sh` | Five-seed mean change −1.09 points | Kubernetes, 1 GPU/run |
| [36-step OPSD](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/opsd-36-step-horizon-seed0) | Longer-horizon robustness; four-seed family | `bash run.sh` | Mean change +1.56 points | Kubernetes, 1 GPU/run, ~2m54s |
| [36-step supported VCSD](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/vcsd-supported-36-step-horizon-seed0) | Longer-horizon headline condition; four-seed family | `bash run.sh` | Mean change −1.17 points | Kubernetes, 1 GPU/run, ~2m59s |
| [36-step unrestricted VCSD](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/vcsd-beta0-36-step-horizon-seed0) | Longer-horizon stability ablation; four-seed family | `bash run.sh` | Mean change −2.15 points | Kubernetes, 1 GPU/run, ~2m59s |
| [Answer-token diagnostic](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/orx/vcsd-ground-truth-answer-token-diagnostic) | Direct original/black/shaped answer-token mass | `bash run.sh` | 11.98% → 29.84% shaped mass | Kubernetes, 1 GPU, 2m38s |

Failed setup branches are omitted from the table except where their repair survives in the final implementation. Raw run IDs and failure lineage remain in `orx exp desc` and the experiment tree.

## Reproduce a condition

Each experiment branch commits its condition in `experiment.json`; the command is unchanged:

```bash
bash run.sh
```

The script fetches public model and dataset assets at run time. It does not redistribute weights or images. The implementation includes the EMA teacher, original-versus-black log-probability contrast, β-relative plausible support, contrast-shaped target, zero-safe forward KL, matched OPSD control, and before/after evaluation.

Key files:

- `run_experiment.py` — training, evaluation, and token diagnostics
- `experiment.json` — bounded protocol and condition
- `.orx/k8s.yaml` — Kubernetes manifest
- `reports/vcsd-reproduction/` — report, figures, and compact data
- `notebooks/vcsd_reproduction.py` — no-training-required interactive walkthrough

The repository also retains legacy files from the original repository snapshot; they are unrelated to this reproduction. Reproduction code is MIT-licensed. Upstream model and dataset licenses continue to govern fetched assets.
