# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
        # Visual Contrastive Self-Distillation: evidence first

        A vision-language model can imitate a slowly updated copy of itself, but
        that teacher may rely on language patterns instead of the image. VCSD
        compares the teacher on the real image and a black image, then emphasizes
        plausible tokens whose probability depends on visual content.

        **Verdict: not reproduced in this bounded setup.** Across four 36-update
        seeds, matched OPSD improved the held-out two-task mean by **+1.56 points**;
        supported VCSD changed it by **−1.17 points**. The proposed token mechanism
        was nevertheless visible: contrast shaping increased ground-truth
        answer-token mass from **11.98% to 29.84%**.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    headline_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 360"
         style="max-width:820px;width:100%;background:#f7f8fb;border-radius:16px">
      <text x="35" y="42" font-size="23" font-weight="700" fill="#172033">
        36-step held-out mean change
      </text>
      <text x="35" y="68" font-size="14" fill="#647087">
        percentage points vs unchanged checkpoint · four seeds
      </text>
      <line x1="430" y1="92" x2="430" y2="320" stroke="#26344f" stroke-width="2"/>
      <g font-size="16" fill="#253047">
        <text x="35" y="130">OPSD</text>
        <text x="35" y="185">VCSD (β=.1)</text>
        <text x="35" y="240">No contrast (α=0)</text>
        <text x="35" y="295">No support (β=0)</text>
      </g>
      <rect x="430" y="108" width="175" height="28" rx="6" fill="#3478f6"/>
      <rect x="299" y="163" width="131" height="28" rx="6" fill="#ef6c5b"/>
      <rect x="343" y="218" width="87" height="28" rx="6" fill="#8d78d6"/>
      <rect x="189" y="273" width="241" height="28" rx="6" fill="#d74c65"/>
      <g font-size="16" font-weight="700">
        <text x="615" y="129" fill="#2462cf">+1.56</text>
        <text x="244" y="184" fill="#b44538">−1.17</text>
        <text x="288" y="239" fill="#6855ad">−0.78</text>
        <text x="203" y="294" fill="#ffffff">−2.15</text>
      </g>
    </svg>
    """
    mo.Html(headline_svg)
    return


@app.cell
def _(mo):
    horizon = mo.ui.dropdown(
        options={"12 updates (5 seeds)": 12, "36 updates (4 seeds)": 36},
        value="36 updates (4 seeds)",
        label="Training horizon",
    )
    horizon
    return (horizon,)


@app.cell
def _(horizon, mo):
    results = {
        12: [
            {"Condition": "Base", "Mean accuracy": 28.91, "Change": 0.00},
            {"Condition": "OPSD", "Mean accuracy": 29.38, "Change": 0.47},
            {"Condition": "VCSD β=.1", "Mean accuracy": 28.44, "Change": -0.47},
            {"Condition": "VCSD β=0", "Mean accuracy": 27.81, "Change": -1.09},
        ],
        36: [
            {"Condition": "Base", "Mean accuracy": 28.91, "Change": 0.00},
            {"Condition": "OPSD", "Mean accuracy": 30.47, "Change": 1.56},
            {"Condition": "VCSD β=.1", "Mean accuracy": 27.73, "Change": -1.17},
            {"Condition": "VCSD α=0", "Mean accuracy": 28.13, "Change": -0.78},
            {"Condition": "VCSD β=0", "Mean accuracy": 26.76, "Change": -2.15},
        ],
    }
    mo.vstack(
        [
            mo.md("## Accuracy: inspect either horizon"),
            horizon,
            mo.ui.table(results[horizon.value], selection=None),
            mo.md(
                "Rows are fixed across conditions; only the rollout seed changes. "
                "Accuracy is the mean of 64 MMStar and 64 MathVista examples."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    token_rows = [
        {"Teacher view / target": "Black image", "Answer-token mass": "1.46%"},
        {"Teacher view / target": "Original image", "Answer-token mass": "11.98%"},
        {"Teacher view / target": "Contrast-shaped", "Answer-token mass": "29.84%"},
    ]
    mo.vstack(
        [
            mo.md(
                r"""
                ## The mechanism, step by step

                For each next token, let the original-image teacher score be
                \( \log p_o \) and the black-image score be \( \log p_b \).
                VCSD forms a visual contrast \( \Delta=\log p_o-\log p_b \),
                keeps tokens within a 10%-relative plausibility support, and
                normalizes \( \log p_o+\Delta \). The student then minimizes
                forward KL to that target while the teacher follows by an
                exponential moving average.

                On 16 fixed MMStar items, the two token variants representing the
                ground-truth choice had the following mean probability mass:
                """
            ),
            mo.ui.table(token_rows, selection=None),
            mo.md(
                "**Observed shaping gain: +17.86 percentage points.** "
                "The answer variants were inside the plausible support on 43.75% "
                "of items, so the effect is selective rather than universal."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    claims = [
        {
            "Claim": "VCSD improves over base and matched OPSD",
            "Observed": "VCSD −1.17; OPSD +1.56 points at 36 updates",
            "Assessment": "Not aligned in this setup",
        },
        {
            "Claim": "Contrast increases image-dependent answer-token mass",
            "Observed": "11.98% → 29.84%",
            "Assessment": "Aligned",
        },
        {
            "Claim": "Removing plausible support destabilizes training",
            "Observed": "β=0 lost 2.15 points vs 1.17; all runs finite",
            "Assessment": "Partially aligned",
        },
    ]
    mo.vstack(
        [
            mo.md("## Claim-by-claim assessment"),
            mo.ui.table(claims, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    scope_markdown = "\n".join(
        [
            "## Scope and provenance",
            "",
            "- **Model:** public `Qwen/Qwen3-VL-2B-Instruct`",
            "- **Training:** 48 fixed public ViRL39K examples; batch one; one rollout; 12 or 36 updates",
            "- **Evaluation:** 64 fixed MMStar and 64 fixed MathVista items",
            "- **Compute:** Kubernetes; NVIDIA RTX PRO 6000 Blackwell Server Edition; 16 GPUs peak; 50m10s (0.84 h) experiment-campaign wall time",
            "- **Largest substitutions:** no official VCSD code, 32-token evaluation cap, strict local answer extraction, and two small benchmarks instead of seven full benchmarks",
            "",
            "The paper’s much higher full-benchmark scores are not directly comparable to these subset numbers. A full reproduction still needs the complete ViRL39K schedule, official prompts and scorers, eight rollouts per prompt, saved checkpoints, and all seven evaluations.",
            "",
            "Read the [full illustrated report](https://github.com/alphaXiv/macil-sd-75f90e4d/blob/main/reports/vcsd-reproduction/report.md) or inspect the [published result data](https://github.com/alphaXiv/macil-sd-75f90e4d/tree/main/reports/vcsd-reproduction/data).",
        ]
    )
    mo.md(scope_markdown)
    return


if __name__ == "__main__":
    app.run()
