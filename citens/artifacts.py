"""Run-artifact renderers beyond the markdown review.

``write_review_browser`` produces a single self-contained ``review_browser.html``
per run (nature-citation's browser-artifact pattern): every claim with its
verdict, the cited references, the ground-text chunk anchors from
provenance.json, and embedded bib/RIS downloads. No server, no build step —
the HTML carries its data and its (vanilla JS) logic in one file, so a run
directory stays the unit of sharing.

The JS lives here as a template string on purpose: a Python package should
not grow a Node toolchain for one static page. When a real web frontend
arrives, it consumes the same JSON artifacts this file reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VERDICT_ZH = {
    "supported": "有依据",
    "partial": "部分依据",
    "background": "仅背景支撑",
    "contradictory": "与原文相左",
    "unsupported": "无依据",
    "unverifiable": "无法核验",
}

_VERDICT_CLASS = {
    "supported": "ok",
    "partial": "warn",
    "background": "warn",
    "contradictory": "bad",
    "unsupported": "bad",
    "unverifiable": "dim",
}


def _load_json(path: Path) -> Any:
    """Best-effort JSON load; None on any failure (missing, corrupt, …)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _b64_data_uri(text: str, mime: str) -> str:
    import base64

    return f"data:{mime};base64," + base64.b64encode(text.encode("utf-8")).decode("ascii")


def write_review_browser(run_dir: str) -> str | None:
    """Render review_browser.html for a finished run; None when the run has
    no verification.json yet (nothing to browse)."""
    d = Path(run_dir)
    ver = _load_json(d / "verification.json")
    if not ver:
        return None

    prov = _load_json(d / "provenance.json") or []
    meta = _load_json(d / "meta.json") or {}
    ground = _load_json(d / "grounding.json") or {}
    leni = _load_json(d / "steps" / "08d_leniency_check.json") or {}
    review_md = ""
    if (d / "review.md").is_file():
        review_md = (d / "review.md").read_text(encoding="utf-8")
    bib = (d / "references.bib").read_text(encoding="utf-8") if (d / "references.bib").is_file() else ""
    ris = (d / "references.ris").read_text(encoding="utf-8") if (d / "references.ris").is_file() else ""

    # provenance is claim-ordered like verification results; join on claim text
    anchors: dict[str, list[dict]] = {}
    for entry in prov:
        if isinstance(entry, dict) and entry.get("evidence_chunks"):
            anchors[entry.get("claim", "")] = entry["evidence_chunks"]

    claims = []
    leni_idx = set(leni.get("claim_indices") or [])
    for i, r in enumerate(ver.get("results", [])):
        claims.append(
            {
                "i": i,
                "text": r.get("claim_text", ""),
                "verdict": r.get("verdict", ""),
                "note": r.get("note", ""),
                "cites": r.get("citation_indices") or [],
                "evidence": anchors.get(r.get("claim_text", ""), []),
                "leni": i in leni_idx,
            }
        )

    counts = {v: ver.get(v, 0) for v in
              ("supported", "partial", "background", "contradictory",
               "unsupported", "unverifiable")}
    papers = [
        {
            "index": p.get("index"),
            "title": p.get("title", ""),
            "fulltext": bool(p.get("has_fulltext")),
            "chunks": p.get("n_chunks", 0),
        }
        for p in ground.get("papers", [])
    ]

    data = {
        "topic": meta.get("topic", d.name),
        "precision": ver.get("citation_precision"),
        "pre": ver.get("pre_rewrite_precision"),
        "post": ver.get("post_rewrite_precision"),
        "counts": counts,
        "claims": claims,
        "papers": papers,
        "papers_cited": ver.get("papers_cited"),
        "papers_total": ver.get("papers_total"),
        "leni": {k: leni.get(k) for k in ("sampled", "downgraded", "agreement_rate")},
        "review_md": review_md,
        "bib_uri": _b64_data_uri(bib, "text/plain") if bib else "",
        "ris_uri": _b64_data_uri(ris, "text/plain") if ris else "",
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    html = _TEMPLATE.replace("__DATA__", payload)
    out = d / "review_browser.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>综述审阅 — citens</title>
<style>
  :root { --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --dim:#6e7781; --line:#d0d7de; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f6f8fa; color: #1f2328; }
  header { padding: 18px 24px; background: #fff; border-bottom: 1px solid var(--line); }
  h1 { font-size: 18px; margin: 0 0 6px; }
  .stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .chip { border: 1px solid var(--line); background: #fff; border-radius: 14px;
          padding: 3px 12px; font-size: 13px; cursor: pointer; user-select: none; }
  .chip.active { background: #1f2328; color: #fff; border-color: #1f2328; }
  .bar { margin: 10px 24px; }
  .bar .track { height: 8px; border-radius: 4px; background: #fff;
                border: 1px solid var(--line); overflow: hidden; display: flex; }
  .bar .seg-ok { background: var(--ok); } .bar .seg-warn { background: #d4a72c; }
  .bar .seg-bad { background: var(--bad); } .bar .seg-dim { background: #c9d1d9; }
  main { display: grid; grid-template-columns: 1fr 340px; gap: 16px; padding: 16px 24px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .claim { background: #fff; border: 1px solid var(--line); border-radius: 8px;
           padding: 12px 14px; margin-bottom: 10px; }
  .claim .head { display: flex; gap: 8px; align-items: baseline; }
  .badge { font-size: 12px; padding: 1px 8px; border-radius: 10px; color: #fff; }
  .badge.ok { background: var(--ok); } .badge.warn { background: var(--warn); }
  .badge.bad { background: var(--bad); } .badge.dim { background: var(--dim); }
  .claim p { margin: 8px 0 4px; line-height: 1.55; }
  .meta { font-size: 12px; color: var(--dim); }
  .note { font-size: 12px; color: #57606a; background: #f6f8fa;
          border-radius: 6px; padding: 6px 8px; margin-top: 6px; }
  details { margin-top: 8px; } summary { cursor: pointer; font-size: 13px; color: #0969da; }
  .ev { font-size: 12px; border-left: 3px solid var(--line); padding: 4px 8px;
        margin: 6px 0; color: #57606a; background: #fafbfc; }
  .ev b { color: #1f2328; }
  input#q { width: 100%; padding: 8px 12px; border: 1px solid var(--line);
            border-radius: 6px; font-size: 14px; margin-bottom: 12px; }
  aside h2 { font-size: 14px; margin: 4px 0 8px; }
  .paper { font-size: 12px; padding: 5px 0; border-bottom: 1px solid #eaeef2; }
  .ft { color: var(--ok); } .noft { color: var(--dim); }
  .dl a { display: inline-block; margin: 4px 8px 4px 0; font-size: 13px; }
  .leni { font-size: 12px; color: var(--warn); }
</style>
</head>
<body>
<header>
  <h1 id="topic"></h1>
  <div class="meta" id="headline"></div>
  <div class="stats" id="stats"></div>
</header>
<div class="bar"><div class="track" id="track"></div></div>
<main>
  <section>
    <input id="q" placeholder="搜索论断 / 引用编号…" />
    <div id="claims"></div>
  </section>
  <aside>
    <h2>论文全文覆盖</h2>
    <div id="papers"></div>
    <h2 style="margin-top:16px">下载</h2>
    <div class="dl" id="dl"></div>
    <div class="leni" id="leni"></div>
  </aside>
</main>
<script>
const DATA = __DATA__;
const ZH = {supported:"有依据",partial:"部分依据",background:"仅背景支撑",
            contradictory:"与原文相左",unsupported:"无依据",unverifiable:"无法核验"};
const CLS = {supported:"ok",partial:"warn",background:"warn",
             contradictory:"bad",unsupported:"bad",unverifiable:"dim"};
let filter = "all";

document.getElementById("topic").textContent = "综述审阅 — " + DATA.topic;
const pct = x => x == null ? "—" : (x*100).toFixed(0) + "%";
document.getElementById("headline").innerHTML =
  "引用精度 <b>" + pct(DATA.precision) + "</b>" +
  (DATA.pre != null ? "（改写前 " + pct(DATA.pre) + " → 改写后 " + pct(DATA.post) + "）" : "") +
  " · 论断 " + DATA.claims.length + " 条" +
  (DATA.papers_cited != null ? " · 引用覆盖 " + DATA.papers_cited + "/" + DATA.papers_total + " 篇" : "");

const stats = document.getElementById("stats");
const mk = (key, label) => {
  const n = key === "all" ? DATA.claims.length : (DATA.counts[key] || 0);
  const el = document.createElement("span");
  el.className = "chip" + (filter === key ? " active" : "");
  el.textContent = label + " " + n;
  el.onclick = () => { filter = key; render(); };
  return el;
};
const chips = () => {
  stats.innerHTML = "";
  stats.appendChild(mk("all", "全部"));
  for (const k of Object.keys(ZH)) stats.appendChild(mk(k, ZH[k]));
};

const track = document.getElementById("track");
const segs = () => {
  track.innerHTML = "";
  const order = ["supported","partial","background","contradictory","unsupported","unverifiable"];
  const cls = {"supported":"seg-ok","partial":"seg-warn","background":"seg-warn",
               "contradictory":"seg-bad","unsupported":"seg-bad","unverifiable":"seg-dim"};
  const total = DATA.claims.length || 1;
  for (const k of order) {
    const n = DATA.counts[k] || 0;
    if (!n) continue;
    const s = document.createElement("div");
    s.className = cls[k];
    s.style.width = (100*n/total) + "%";
    s.title = ZH[k] + " " + n;
    track.appendChild(s);
  }
};

function esc(s){const d=document.createElement("div");d.textContent=s==null?"":s;return d.innerHTML;}

function render() {
  chips(); segs();
  const q = document.getElementById("q").value.trim().toLowerCase();
  const box = document.getElementById("claims");
  box.innerHTML = "";
  let shown = 0;
  for (const c of DATA.claims) {
    if (filter !== "all" && c.verdict !== filter) continue;
    if (q && !(c.text.toLowerCase().includes(q) ||
               c.cites.some(i => ("["+i+"]").includes(q)))) continue;
    if (++shown > 300) break;
    const el = document.createElement("div");
    el.className = "claim";
    let html = '<div class="head"><span class="badge ' + CLS[c.verdict] + '">' +
      ZH[c.verdict] + '</span><span class="meta">论断 ' + (c.i+1) +
      (c.leni ? ' · <span class="leni">★严格复审建议降级</span>' : '') +
      ' · 引用 ' + c.cites.map(i => "["+i+"]").join(" ") + '</span></div>';
    html += "<p>" + esc(c.text) + "</p>";
    if (c.note) html += '<div class="note">核验理由：' + esc(c.note) + "</div>";
    if (c.evidence && c.evidence.length) {
      html += '<details><summary>证据锚点（' + c.evidence.length + "）</summary>";
      for (const e of c.evidence) {
        html += '<div class="ev"><b>[' + e.index + "] " + e.chunk_id + " (" + e.kind +
                ")</b><br>" + esc(e.excerpt) + "</div>";
      }
      html += "</details>";
    }
    el.innerHTML = html;
    box.appendChild(el);
  }
  if (!shown) box.innerHTML = '<div class="meta">无匹配论断</div>';
}
document.getElementById("q").addEventListener("input", render);

const papers = document.getElementById("papers");
let ft = 0;
for (const p of DATA.papers) {
  if (p.fulltext) ft++;
  const d = document.createElement("div");
  d.className = "paper";
  d.innerHTML = "[" + p.index + "] " + esc(p.title.length > 46 ? p.title.slice(0,46)+"…" : p.title) +
    ' <span class="' + (p.fulltext ? "ft" : "noft") + '">' +
    (p.fulltext ? "全文·" + p.chunks + "块" : "仅摘要") + "</span>";
  papers.appendChild(d);
}
if (DATA.papers.length)
  papers.insertAdjacentHTML("afterbegin", '<div class="meta">全文 ' + ft + "/" +
    DATA.papers.length + " 篇</div>");

const dl = document.getElementById("dl");
dl.innerHTML = "";
if (DATA.review_md) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([DATA.review_md], {type:"text/markdown"}));
  a.download = "review.md"; a.textContent = "review.md";
  dl.appendChild(a);
}
for (const [uri, name] of [[DATA.bib_uri,"references.bib"],[DATA.ris_uri,"references.ris"]]) {
  if (!uri) continue;
  const a = document.createElement("a");
  a.href = uri; a.download = name; a.textContent = name;
  dl.appendChild(a);
}
if (DATA.leni && DATA.leni.sampled) {
  document.getElementById("leni").textContent =
    "宽严抽检：采样 " + DATA.leni.sampled + " 条 supported，" +
    DATA.leni.downgraded + " 条建议降级（一致率 " +
    (DATA.leni.agreement_rate == null ? "—" : (DATA.leni.agreement_rate*100).toFixed(0)+"%") + "）";
}
render();
</script>
</body>
</html>
"""
