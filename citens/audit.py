"""Human-audit tooling: the calibration loop for the verifier.

`generate_audit_sheet` renders every claim with its machine verdict, cited
papers, and a fill-in slot; a human judges s/p/u. `ingest_audit` parses the
filled sheet, computes agreement with the machine verdicts, and writes
`audit_result.json` — the leniency signal that calibrates how much the
reported citation precision can be trusted (and later, the verifier prompt).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_VERDICTS = {"s": "supported", "p": "partial", "u": "unsupported"}

# machine background/contradictory both mean "not grounded by this citation"
# for a human filling s/p/u; normalize before comparing, keep raw for confusion
_HUMAN_EQUIV = {"background": "unsupported", "contradictory": "unsupported"}


def _refs_from_bib(run_dir: str) -> list[str]:
    """Reference labels in CitationTable order (references.bib entry order)."""
    bib = (Path(run_dir) / "references.bib").read_text(encoding="utf-8")
    entries = re.findall(r"@(\w+)\{[^,]*,\s*(.*?)\n\}", bib, re.S)
    refs: list[str] = []
    for kind, body in entries:
        def field(name: str, b: str = body) -> str:
            m = re.search(rf"{name}\s*=\s*{{(.*?)}}", b, re.S)
            return (m.group(1).strip()[:70] if m else "")

        year = re.search(r"year\s*=\s*{(\d{4})", body)
        venue = field("journal") or field("booktitle") or kind
        refs.append(f"[{len(refs)}] {field('title')} ({venue}, {year.group(1) if year else '?'})")
    return refs


def generate_audit_sheet(run_dir: str, out_name: str = "审核清单.md") -> str:
    """Render the per-claim audit sheet for a completed run."""
    ver = json.loads((Path(run_dir) / "verification.json").read_text(encoding="utf-8"))
    refs = _refs_from_bib(run_dir)
    leni_path = Path(run_dir) / "steps" / "08d_leniency_check.json"
    leni = (
        json.loads(leni_path.read_text(encoding="utf-8"))
        if leni_path.is_file()
        else {}
    )

    lines = [
        f"# 人工审核清单 — {Path(run_dir).name}\n",
        f"机器自报精度: {ver['citation_precision'] * 100:.0f}%  "
        f"(改写前 {ver.get('pre_rewrite_precision', '-')}, "
        f"改写后 {ver.get('post_rewrite_precision', '-')})  "
        f"| 总论断 {ver['total_claims']} 条: supported {ver['supported']} / "
        f"partial {ver['partial']} / background {ver.get('background', 0)} / "
        f"contradictory {ver.get('contradictory', 0)} / "
        f"unsupported {ver['unsupported']} / "
        f"unverifiable {ver['unverifiable']}",
    ]
    if leni.get("sampled"):
        lines.append(
            f"宽严抽检: 采样 {leni['sampled']} 条 supported, "
            f"{leni['downgraded']} 条建议降级 (一致率 {leni['agreement_rate']})"
        )
    lines += [
        "\n判定标准: supported(s)=被引文献确有此依据; "
        "partial(p)=大方向对但有夸大/无依据细节; unsupported(u)=被引文献没有此依据"
        "（机器的 background=只支撑背景 / contradictory=与原文相左，均按 u 对待）",
        "用法: 每条论断下'人工判定:'后填 s/p/u, 然后运行 "
        "`citens audit <run目录> --ingest 审核清单.md`\n",
        "## 参考文献索引\n",
    ]
    lines += [f"- {r}" for r in refs]
    lines.append("")
    for i, r in enumerate(ver["results"]):
        flag = " ★严格复审建议降级" if i in leni.get("claim_indices", []) else ""
        lines.append(f"## 论断 {i + 1} — 机器判定: {r['verdict']}{flag}")
        lines.append(f"> {r['claim_text']}\n")
        for n in r.get("citation_indices") or []:
            if 0 <= n < len(refs):
                lines.append(f"- 被引: {refs[n]}")
        lines.append(f"- 机器理由: {(r.get('note') or '')[:150]}")
        lines.append("- 人工判定: ____\n")

    out = Path(run_dir) / out_name
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def ingest_audit(run_dir: str, sheet_path: str) -> dict:
    """Parse a filled audit sheet; compute human-vs-machine agreement."""
    ver = json.loads((Path(run_dir) / "verification.json").read_text(encoding="utf-8"))
    sheet = Path(sheet_path).read_text(encoding="utf-8")

    judged: dict[int, str] = {}
    for m in re.finditer(
        r"## 论断 (\d+) —.*?\n.*?(?:\n.*?)??- 人工判定:\s*([spuSPU])",
        sheet,
        re.S,
    ):
        judged[int(m.group(1)) - 1] = _VERDICTS[m.group(2).lower()]

    results = ver["results"]
    matched = [(i, v) for i, v in judged.items() if 0 <= i < len(results)]
    if not matched:
        raise ValueError(
            "no '人工判定: s/p/u' entries found — fill the sheet before ingesting"
        )

    exact = sum(
        1
        for i, v in matched
        if _HUMAN_EQUIV.get(results[i]["verdict"], results[i]["verdict"]) == v
    )
    # a one-step leniency gap: machine supported vs human partial, or
    # machine partial vs human unsupported (background/contradictory count as
    # unsupported — they are both "not grounded by this citation")
    order = {"supported": 0, "partial": 1, "unsupported": 2}

    def _m(v: str) -> int:
        return order.get(_HUMAN_EQUIV.get(v, v), 1)

    lenient = sum(1 for i, v in matched if _m(results[i]["verdict"]) < order.get(v, 1))
    strict = sum(1 for i, v in matched if _m(results[i]["verdict"]) > order.get(v, 1))
    # human-grounded precision over the judged sample (the honest headline)
    human_ok = sum(1 for _i, v in matched if v in ("supported", "partial"))

    confusion: dict[str, dict[str, int]] = {}
    for i, v in matched:
        confusion.setdefault(results[i]["verdict"], {}).setdefault(v, 0)
        confusion[results[i]["verdict"]][v] += 1

    report = {
        "judged": len(matched),
        "of_total": len(results),
        "agreement_rate": round(exact / len(matched), 3),
        "machine_lenient": lenient,
        "machine_strict": strict,
        "human_grounded_rate": round(human_ok / len(matched), 3),
        "machine_reported_precision": ver["citation_precision"],
        "confusion_machine_to_human": confusion,
        "per_claim": [
            {
                "claim_index": i,
                "machine": results[i]["verdict"],
                "human": v,
                "claim_text": results[i]["claim_text"],
            }
            for i, v in matched
        ],
    }
    out = Path(run_dir) / "audit_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
