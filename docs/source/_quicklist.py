from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypair.biserial import Biserial
from pypair.contingency import AgreementTable, BinaryTable, CategoricalTable, ConfusionMatrix
from pypair.continuous import Concordance, Continuous, CorrelationRatio


@dataclass(frozen=True)
class MeasureGroup:
    title: str
    cls: type[Any]
    fallback_note: str


GROUPS = [
    MeasureGroup(
        title="Binary-Binary",
        cls=BinaryTable,
        fallback_note="Binary-binary coefficient derived from the 2x2 contingency cells a, b, c, d and total n.",
    ),
    MeasureGroup(
        title="Confusion Matrix, Binary-Binary",
        cls=ConfusionMatrix,
        fallback_note="Confusion-matrix metric derived from the TP, FN, FP, TN counts.",
    ),
    MeasureGroup(
        title="Categorical-Categorical",
        cls=CategoricalTable,
        fallback_note="Categorical association statistic derived from contingency-table counts and marginals.",
    ),
    MeasureGroup(
        title="Agreement, Categorical-Categorical",
        cls=AgreementTable,
        fallback_note="Agreement statistic computed from a square contingency table of paired ratings or labels.",
    ),
    MeasureGroup(
        title="Binary-Continuous, Biserial",
        cls=Biserial,
        fallback_note="Binary-continuous association derived from the group means y_0, y_1, proportions p, q, and sample standard deviation sigma.",
    ),
    MeasureGroup(
        title="Categorical-Continuous",
        cls=CorrelationRatio,
        fallback_note="Categorical-continuous statistic computed across category-level samples of a numeric response.",
    ),
    MeasureGroup(
        title="Ordinal-Ordinal, Concordance",
        cls=Concordance,
        fallback_note="Concordance statistic derived from concordant, discordant, and tied observation pairs.",
    ),
    MeasureGroup(
        title="Continuous-Continuous",
        cls=Continuous,
        fallback_note="Continuous-continuous association statistic returned by the delegated SciPy computation.",
    ),
]

PURE_MATH_LINE = re.compile(r"^:math:`[^`]+`[.,;:]?$")


OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("BinaryTable", "adjusted_rand_index"): {
        "equations": [
            r":math:`ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \frac{\left(\sum_i \binom{a_i}{2}\right)\left(\sum_j \binom{b_j}{2}\right)}{\binom{n}{2}}}{\frac{1}{2}\left(\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right) - \frac{\left(\sum_i \binom{a_i}{2}\right)\left(\sum_j \binom{b_j}{2}\right)}{\binom{n}{2}}}`",
        ],
        "note": "Adjusted Rand index for the induced contingency table; values can be negative and the binomial terms can overflow for large n.",
    },
    ("BinaryTable", "contingency_coefficient"): {
        "equations": [r":math:`C = \sqrt{\frac{\chi^2}{n + \chi^2}}`"],
        "note": "Contingency coefficient computed from the chi-square statistic of the binary table.",
    },
    ("BinaryTable", "chisq_dof"): {
        "equations": [r":math:`(R - 1)(C - 1)`"],
        "note": "Degrees of freedom for the chi-square statistic on the induced contingency table.",
    },
    ("BinaryTable", "cramer_v"): {
        "equations": [r":math:`V = \sqrt{\frac{\chi^2}{n}}`"],
        "note": "Cramer's V as implemented for the binary 2x2 case.",
    },
    ("BinaryTable", "mcnemar_test"): {
        "equations": [
            r":math:`\chi^2 = \frac{(b-c)^2}{b+c}`",
            r":math:`p = 1 - F_{\chi^2_1}(\chi^2)`",
        ],
        "note": "McNemar's chi-square test on the off-diagonal disagreement counts.",
    },
    ("BinaryTable", "odds_ratio"): {
        "equations": [r":math:`OR = \frac{p_{11}p_{00}}{p_{10}p_{01}} = \frac{ad}{bc}`"],
        "note": "Odds ratio, also referred to in the code as the cross-product ratio.",
    },
    ("BinaryTable", "tanimoto_distance"): {
        "equations": [r":math:`D_T = -\log_2(\mathrm{roger\_tanimoto})`"],
        "note": "Distance form derived directly from the Roger-Tanimoto similarity in the current implementation.",
    },
    ("BinaryTable", "tschuprow_t"): {
        "equations": [r":math:`T = \sqrt{\chi^2}`"],
        "note": "Current implementation returns the square root of the chi-square statistic for the binary table.",
    },
    ("BinaryTable", "uncertainty_coefficient_reversed"): {
        "equations": [r":math:`U_\mathrm{rev} = \frac{I(X;Y)}{H(\mathrm{rows})}`"],
        "note": "Mutual information normalized by the row entropy, reversing the default direction used by uncertainty_coefficient.",
    },
    ("CategoricalTable", "adjusted_rand_index"): {
        "equations": [
            r":math:`ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - \frac{\left(\sum_i \binom{a_i}{2}\right)\left(\sum_j \binom{b_j}{2}\right)}{\binom{n}{2}}}{\frac{1}{2}\left(\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right) - \frac{\left(\sum_i \binom{a_i}{2}\right)\left(\sum_j \binom{b_j}{2}\right)}{\binom{n}{2}}}`",
        ],
        "note": "Adjusted Rand index for the categorical contingency table; values can be negative and the binomial terms can overflow for large n.",
    },
    ("CategoricalTable", "chisq_dof"): {
        "equations": [r":math:`(R - 1)(C - 1)`"],
        "note": "Degrees of freedom for the chi-square statistic on the induced contingency table.",
    },
    ("CategoricalTable", "uncertainty_coefficient_reversed"): {
        "equations": [r":math:`U_\mathrm{rev} = \frac{I(X;Y)}{H(\mathrm{rows})}`"],
        "note": "Mutual information normalized by the row entropy, reversing the default direction used by uncertainty_coefficient.",
    },
    ("BinaryTable", "gk_lambda_reversed"): {
        "equations": [r":math:`\lambda_{A|B} = \frac{\sum_c \max_r N_{rc} - \max_r N_{r*}}{N - \max_r N_{r*}}`"],
        "note": "Reverse-direction Goodman-Kruskal lambda, predicting rows from columns.",
    },
    ("CategoricalTable", "gk_lambda_reversed"): {
        "equations": [r":math:`\lambda_{A|B} = \frac{\sum_c \max_r N_{rc} - \max_r N_{r*}}{N - \max_r N_{r*}}`"],
        "note": "Reverse-direction Goodman-Kruskal lambda, predicting rows from columns.",
    },
    ("ConfusionMatrix", "tp"): {
        "equations": [r":math:`TP`"],
        "note": "Raw true-positive count from the confusion matrix.",
    },
    ("ConfusionMatrix", "fn"): {
        "equations": [r":math:`FN`"],
        "note": "Raw false-negative count from the confusion matrix.",
    },
    ("ConfusionMatrix", "fp"): {
        "equations": [r":math:`FP`"],
        "note": "Raw false-positive count from the confusion matrix.",
    },
    ("ConfusionMatrix", "tn"): {
        "equations": [r":math:`TN`"],
        "note": "Raw true-negative count from the confusion matrix.",
    },
    ("ConfusionMatrix", "precision"): {
        "equations": [r":math:`PPV = \frac{TP}{TP + FP}`"],
        "note": "Alias of positive predictive value (PPV).",
    },
    ("ConfusionMatrix", "recall"): {
        "equations": [r":math:`TPR = \frac{TP}{TP + FN}`"],
        "note": "Alias of true positive rate (TPR).",
    },
    ("ConfusionMatrix", "sensitivity"): {
        "equations": [r":math:`TPR = \frac{TP}{TP + FN}`"],
        "note": "Alias of true positive rate (TPR).",
    },
    ("ConfusionMatrix", "specificity"): {
        "equations": [r":math:`TNR = \frac{TN}{TN + FP}`"],
        "note": "Alias of true negative rate (TNR).",
    },
    ("Biserial", "biserial"): {
        "equations": [
            r":math:`r_{pb} = \frac{(y_1 - y_0)\sqrt{pq}}{\sigma}`",
            r":math:`r_b = r_{pb}\frac{\sqrt{pq}}{\phi(\Phi^{-1}(q))}`",
        ],
        "note": "Biserial correlation using the point-biserial term with the standard normal PDF/CDF correction from the implementation.",
    },
    ("Biserial", "point_biserial"): {
        "equations": [r":math:`r_{pb} = \frac{(y_1 - y_0)\sqrt{pq}}{\sigma}`"],
        "note": "Point-biserial correlation between a binary variable and a continuous response.",
    },
    ("Biserial", "rank_biserial"): {
        "equations": [r":math:`r_{rb} = \frac{2(y_1 - y_0)}{n}`"],
        "note": "Rank-biserial statistic as currently implemented from the two group means and sample size.",
    },
    ("CorrelationRatio", "anova"): {
        "equations": [r":math:`F = \frac{SS_B / (k - 1)}{SS_W / (n - k)}`"],
        "note": "One-way ANOVA F statistic with p-value returned by scipy.stats.f_oneway.",
    },
    ("CorrelationRatio", "eta"): {
        "equations": [r":math:`\eta = \sqrt{\eta^2}`"],
        "note": "Correlation ratio magnitude derived as the square root of eta_squared.",
    },
    ("CorrelationRatio", "calinski_harabasz"): {
        "equations": [r":math:`CH = \frac{\operatorname{tr}(B_k)}{\operatorname{tr}(W_k)} \cdot \frac{n-k}{k-1}`"],
        "note": "Calinski-Harabasz separation score from scikit-learn over the grouped continuous values.",
    },
    ("CorrelationRatio", "davies_bouldin"): {
        "equations": [r":math:`DB = \frac{1}{k}\sum_i \max_{j \neq i}\frac{s_i + s_j}{d_{ij}}`"],
        "note": "Davies-Bouldin index from scikit-learn over category-labelled one-dimensional samples.",
    },
    ("CorrelationRatio", "kruskal"): {
        "equations": [r":math:`H = \frac{12}{N(N+1)}\sum_i \frac{R_i^2}{n_i} - 3(N+1)`"],
        "note": "Kruskal-Wallis H statistic with p-value returned by scipy.stats.kruskal.",
    },
    ("CorrelationRatio", "silhouette"): {
        "equations": [r":math:`s_i = \frac{b_i - a_i}{\max(a_i, b_i)}`"],
        "note": "Silhouette coefficient over category-labelled one-dimensional samples; the implementation returns the dataset average.",
    },
    ("Continuous", "kendall"): {
        "equations": [r":math:`\tau = \frac{C-D}{\sqrt{(C+D+T_x)(C+D+T_y)}}`"],
        "note": "Kendall rank correlation and p-value returned by scipy.stats.kendalltau.",
    },
    ("Continuous", "pearson"): {
        "equations": [
            r":math:`r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2 \sum_i (y_i-\bar{y})^2}}`"
        ],
        "note": "Pearson linear correlation and p-value returned by scipy.stats.pearsonr.",
    },
    ("Continuous", "regression"): {
        "equations": [r":math:`y = \beta_0 + \beta_1 x`"],
        "note": "Linear regression via scipy.stats.linregress; this API returns the correlation coefficient r and p-value.",
    },
    ("Continuous", "spearman"): {
        "equations": [r":math:`\rho = \mathrm{corr}(\mathrm{rank}(x), \mathrm{rank}(y))`"],
        "note": "Spearman rank correlation and p-value returned by scipy.stats.spearmanr.",
    },
}


def _unique_measures(measures: list[str]) -> list[str]:
    return list(dict.fromkeys(measures))


def _is_title_line(line: str) -> bool:
    text = line.strip().strip(".")
    if not text:
        return False
    if ":math:`" in text or "<" in text or "http" in text:
        return False
    tokens = text.replace("`", "").replace("_", " ").split()
    lowered = {token.lower() for token in tokens}
    if lowered & {"is", "are", "for", "with", "alias", "computes", "returns"}:
        return False
    return len(tokens) <= 4 and len(text) <= 48


def _extract_equations(doc: str) -> list[str]:
    equations: list[str] = []
    for raw_line in doc.splitlines():
        line = raw_line.strip()
        if line.startswith(":return") or line.startswith(":returns"):
            continue
        if not line:
            continue

        bullet_stripped = line[2:].strip() if line.startswith("- ") else line
        if PURE_MATH_LINE.match(bullet_stripped):
            equations.append(bullet_stripped)
            continue

        math_spans = re.findall(r":math:`([^`]+)`", line)
        if not math_spans:
            continue

        for span in math_spans:
            if "=" in span:
                equations.append(f":math:`{span}`")

    return list(dict.fromkeys(equations))


def _extract_note(doc: str, fallback_note: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in doc.splitlines() if line.strip()]

    while lines and _is_title_line(lines[0]):
        lines.pop(0)

    note_lines: list[str] = []
    for line in lines:
        if line in {"Aliases", "References", "Where"}:
            break
        if line.startswith(":return") or line.startswith(":returns"):
            break
        bullet_stripped = line[2:].strip() if line.startswith("- ") else line
        if PURE_MATH_LINE.match(bullet_stripped):
            break
        if line.startswith("- "):
            continue
        note_lines.append(line)

    note = " ".join(note_lines).strip()
    if note:
        note = re.split(r"(?<=[.!?])\s+", note)[0].strip()
    return note or fallback_note


def _measure_doc(cls: type[Any], measure: str) -> str:
    prop = getattr(cls, measure)
    if not isinstance(prop, property):
        return ""
    return inspect.getdoc(prop.fget) or ""


def _measure_payload(group: MeasureGroup, measure: str) -> tuple[list[str], str]:
    override = OVERRIDES.get((group.cls.__name__, measure))
    if override is not None:
        return override["equations"], override["note"]

    doc = _measure_doc(group.cls, measure)
    equations = _extract_equations(doc)
    note = _extract_note(doc, group.fallback_note)
    if not equations:
        raise ValueError(f"Missing equation for {group.cls.__name__}.{measure}")
    return equations, note


def _render_table(group: MeasureGroup) -> list[str]:
    measures = _unique_measures(group.cls.measures())
    title = f"{group.title} ({len(measures)})"
    lines = [title, "-" * len(title), "", ".. list-table::", "   :header-rows: 1", "   :widths: 18 44 38", ""]
    lines.extend(["   * - Name", "     - Equation", "     - Note"])

    for measure in measures:
        equations, note = _measure_payload(group, measure)
        lines.append(f"   * - ``{measure}``")
        lines.append(f"     - | {equations[0]}")
        for equation in equations[1:]:
            lines.append(f"       | {equation}")
        lines.append(f"     - {note}")

    lines.append("")
    return lines


def render_quicklist_tables() -> str:
    lines = [
        ".. This file is generated by docs/source/_quicklist.py. Edit the generator instead of hand-editing this table.",
        "",
    ]
    for group in GROUPS:
        lines.extend(_render_table(group))
    return "\n".join(lines).rstrip() + "\n"


def generate_quicklist_tables(source_dir: str | Path) -> Path:
    source_path = Path(source_dir)
    output_dir = source_path / "_generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quicklist_tables.rst"
    output_path.write_text(render_quicklist_tables(), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    generate_quicklist_tables(Path(__file__).resolve().parent)
