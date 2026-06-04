"""Render structured insights as ranked natural-language sentences.

The output is capped at ``MAX_SENTENCES`` items, ordered by importance:
trends -> correlations -> category breakdowns -> distribution flags.
"""

from __future__ import annotations


MAX_SENTENCES = 10
HIGH_SKEW_THRESHOLD = 1.0
OUTLIER_FLAG_RATIO = 0.05


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _trend_sentence(trend: dict) -> str:
    direction_word = "increased" if trend["direction"] == "up" else "decreased"
    pct = abs(trend["pct_change"])
    col = trend["column"]
    first = trend["first_period"]
    last = trend["last_period"]
    return f"{col} {direction_word} {_fmt_pct(pct)} from {first} to {last}."


def _correlation_sentence(corr: dict) -> str:
    strength = "strongly" if abs(corr["r"]) >= 0.7 else "moderately"
    sign = "positively" if corr["r"] >= 0 else "negatively"
    return (
        f"{corr['a']} and {corr['b']} are {strength} {sign} correlated "
        f"(r={corr['r']:.2f})."
    )


def _category_sentence(item: dict) -> str:
    cat = item["category_col"]
    num = item["numeric_col"]
    top_v = item["top"]["value"]
    bot_v = item["bottom"]["value"]
    pct = abs(item["pct_diff"])
    if item["pct_diff"] >= 0:
        return f"{cat} '{top_v}' has {_fmt_pct(pct)} higher average {num} than '{bot_v}'."
    return f"{cat} '{top_v}' has {_fmt_pct(pct)} lower average {num} than '{bot_v}'."


def _distribution_sentence(col: str, stats: dict) -> str | None:
    parts: list[str] = []
    skew = stats.get("skew")
    if skew is not None and abs(skew) > HIGH_SKEW_THRESHOLD:
        direction = "right" if skew > 0 else "left"
        parts.append(f"{direction}-skewed (skew={skew:.2f})")
    count = stats.get("count") or 0
    outliers = stats.get("outliers_iqr") or 0
    if count > 0 and outliers / count > OUTLIER_FLAG_RATIO:
        parts.append(f"{outliers} IQR-outlier values ({outliers / count:.0%})")
    if not parts:
        return None
    return f"{col} is " + "; ".join(parts) + "."


def render_sentences(
    trends: list[dict],
    correlations: list[dict],
    category_breakdowns: list[dict],
    column_stats: dict[str, dict],
) -> list[str]:
    sentences: list[str] = []
    for trend in trends:
        sentences.append(_trend_sentence(trend))
        if len(sentences) >= MAX_SENTENCES:
            return sentences

    for corr in correlations:
        sentences.append(_correlation_sentence(corr))
        if len(sentences) >= MAX_SENTENCES:
            return sentences

    for item in category_breakdowns:
        sentences.append(_category_sentence(item))
        if len(sentences) >= MAX_SENTENCES:
            return sentences

    for col, payload in column_stats.items():
        if payload.get("type") not in {"integer", "float"}:
            continue
        sentence = _distribution_sentence(col, payload.get("stats", {}))
        if sentence:
            sentences.append(sentence)
            if len(sentences) >= MAX_SENTENCES:
                return sentences

    return sentences
