# Visual Contrastive Self-Distillation — bounded reproduction

This public repository contains an independent, claim-oriented reproduction of
[Visual Contrastive Self-Distillation](https://arxiv.org/abs/2607.21556) on
Qwen3-VL-2B-Instruct. The live result, report, figures, and tutorial notebook
will be published here after the fresh Kubernetes runs finish.

The experiment uses a fixed public ViRL39K subset and fixed public MMStar and
MathVista subsets. It compares the unchanged checkpoint, compute-matched
answer-hint OPSD, VCSD with relative plausible support, and the paper's
unrestricted-support ablation. All conditions share the same run command.

## Reproduction command

```bash
bash run.sh
```

The condition is committed in `experiment.json`; the command never changes
between experiment-tree nodes.

## Repository layout

- `run_experiment.py`: Qwen3-VL training, evaluation, and diagnostics
- `experiment.json`: experiment condition and bounded protocol
- `.orx/k8s.yaml`: four-GPU Kubernetes job
- `reports/vcsd-reproduction/`: final reader-facing report and figures
- `notebooks/vcsd_reproduction.py`: self-contained marimo walkthrough

## License

Reproduction code is released under the MIT License. Model and dataset assets
are fetched at run time from their public upstream repositories and are not
redistributed here.
