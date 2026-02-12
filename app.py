from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.recommend import parse_task_profile, recommend, _extract_budget_from_text

app = FastAPI(title="LLM Recommender")

class RecommendRequest(BaseModel):
    task_text: str
    topk: int = 5
    max_price_per_1m: float | None = None
    min_context: int | None = None
    provider_allowlist: str | None = None
    missing_policy: str = "penalize"  # "neutral" or "penalize"
    refresh: bool = False
    max_age_hours: float = 24.0

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    # Simple UI + pretty JSON + cards for recommendations
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>LLM Recommender</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:32px auto;padding:0 16px}
    textarea,input,select{width:100%;padding:10px;border:1px solid #ddd;border-radius:10px;font-size:14px}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
    button{padding:10px 14px;border:0;border-radius:10px;font-weight:600;cursor:pointer}
    button.primary{background:#111;color:#fff}
    .card{border:1px solid #eee;border-radius:16px;padding:14px;margin:12px 0;box-shadow:0 2px 10px rgba(0,0,0,.04)}
    .muted{color:#666;font-size:13px}
    pre{background:#0b1020;color:#e7e7e7;padding:14px;border-radius:14px;overflow:auto}
    .pill{display:inline-block;background:#f2f2f2;border-radius:999px;padding:4px 10px;font-size:12px;margin-right:6px}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    @media(max-width:820px){.row,.row3,.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <h1>LLM Recommender</h1>
  <p class="muted">Enter a prompt, get ranked models + metrics.</p>

  <label>Task prompt</label>
  <textarea id="task" rows="3">python debugging, prefer quality, budget $5/1M tokens</textarea>

  <div class="row3" style="margin-top:12px">
    <div>
      <label>Top K</label>
      <input id="topk" type="number" value="5" min="1" max="20"/>
    </div>
    <div>
      <label>Missing policy</label>
      <select id="missing_policy">
        <option value="penalize" selected>penalize</option>
        <option value="neutral">neutral</option>
      </select>
    </div>
    <div>
      <label>Max age (hours)</label>
      <input id="max_age_hours" type="number" value="24" min="0" step="0.5"/>
    </div>
  </div>

  <div class="row" style="margin-top:12px">
    <div>
      <label>Max price per 1M (optional)</label>
      <input id="max_price" type="number" placeholder="e.g., 5" step="0.01"/>
    </div>
    <div>
      <label>Provider allowlist (optional)</label>
      <input id="providers" placeholder="e.g., OpenAI,Google"/>
    </div>
  </div>

  <div style="margin-top:12px">
    <label><input id="refresh" type="checkbox"/> Force refresh before recommend</label>
  </div>

  <div style="margin-top:12px">
    <button class="primary" onclick="run()">Recommend</button>
  </div>

  <h2 style="margin-top:22px">Top Recommendations</h2>
  <div id="cards"></div>

  <h2>Raw JSON</h2>
  <pre id="out">{}</pre>

<script>
async function run(){
  const payload = {
    task_text: document.getElementById('task').value,
    topk: parseInt(document.getElementById('topk').value || "5", 10),
    max_price_per_1m: document.getElementById('max_price').value ? parseFloat(document.getElementById('max_price').value) : null,
    provider_allowlist: document.getElementById('providers').value || null,
    missing_policy: document.getElementById('missing_policy').value,
    refresh: document.getElementById('refresh').checked,
    max_age_hours: parseFloat(document.getElementById('max_age_hours').value || "24"),
  };

  const res = await fetch('/api/recommend', {
    method:'POST',
    headers:{'content-type':'application/json'},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);

  const cards = document.getElementById('cards');
  cards.innerHTML = "";

  if (data.warning){
    const w = document.createElement('div');
    w.className = "card";
    w.innerHTML = `<div class="pill">warning</div><div>${escapeHtml(data.warning)}</div>`;
    cards.appendChild(w);
  }

  (data.recommendations || []).forEach(r => {
    const c = document.createElement('div');
    c.className = "card";
    const m = r.metrics || {};
    c.innerHTML = `
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start">
        <div>
          <div style="font-weight:700;font-size:16px">${escapeHtml(r.model_name || "")}</div>
          <div class="muted">${escapeHtml(r.provider || "")} • <span class="pill">${escapeHtml(r.canonical_model_key || "")}</span></div>
        </div>
        <div style="text-align:right">
          <div style="font-weight:800;font-size:18px">${Number(r.score).toFixed(4)}</div>
          <div class="muted">score</div>
        </div>
      </div>
      <div class="grid" style="margin-top:10px">
        <div><div class="muted">quality</div><div>${fmt(m.quality_metric)}</div></div>
        <div><div class="muted">price / 1M</div><div>${fmt(m.price_input_per_1m)}</div></div>
        <div><div class="muted">out tok/s</div><div>${fmt(m.output_tokens_per_s)}</div></div>
        <div><div class="muted">context</div><div>${fmt(m.context_window)}</div></div>
      </div>
      <div class="muted" style="margin-top:10px">${escapeHtml(r.justification || "")}</div>
    `;
    cards.appendChild(c);
  });
}

function fmt(v){
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}
function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>
"""


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest) -> dict:
    max_price = req.max_price_per_1m
    if max_price is None:
        max_price = _extract_budget_from_text(req.task_text)

    profile = parse_task_profile(
        task_text=req.task_text,
        max_price_per_1m=max_price,
        min_context=req.min_context,
        provider_allowlist=req.provider_allowlist,
    )

    result = recommend(
        profile,
        topk=req.topk,
        missing_policy=req.missing_policy,
        refresh=req.refresh,
        max_age_hours=req.max_age_hours,
    )
    return result
