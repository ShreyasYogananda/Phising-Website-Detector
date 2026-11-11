/* PhishGuard front end. All dynamic text goes through textContent / el() so nothing from a URL or API
   response is ever interpreted as HTML. */
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("data-")) n.setAttribute(k, v);
      else n[k] = v;
    }
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
}
const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const NONE = (s) => !s || s.startsWith("(");

/* ---------- Tabs ---------- */
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.view === name)));
  document.querySelectorAll("[role=tabpanel]").forEach((p) => { p.hidden = p.id !== "v-" + name; });
  if (name === "eval") loadEvaluation();
  try { history.replaceState(null, "", "#" + name); } catch (_) { /* ignore */ }
}

function initTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => showView(t.dataset.view));
    t.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      const next = tabs[(i + d + tabs.length) % tabs.length];
      next.focus();
      showView(next.dataset.view);
    });
  });
  const start = location.hash.slice(1);
  if (["eval", "how"].includes(start)) showView(start);
}

/* ---------- Check form ---------- */
function initForm() {
  const form = $("check-form");
  const input = $("url-input");
  const btn = $("check-btn");
  const err = $("form-error");

  document.querySelectorAll(".chip").forEach((c) =>
    c.addEventListener("click", () => {
      input.value = c.dataset.url;
      form.requestSubmit();
    })
  );

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.textContent = "";
    const url = input.value.trim();
    if (!url) { err.textContent = "Paste a web address to check."; input.focus(); return; }

    btn.disabled = true;
    clear(btn).append(el("span", { class: "spinner" }), "Checking");
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, enable_page_analysis: $("page-check").checked }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { err.textContent = data.error || "Could not check that link. Try again."; return; }
      renderResult(data);
    } catch (_) {
      err.textContent = "Could not reach the server. Check that PhishGuard is still running, then try again.";
    } finally {
      btn.disabled = false;
      btn.textContent = "Check link";
    }
  });
}

/* ---------- Result ---------- */
const TONE = { danger: "danger", warning: "warning", success: "safe" };

function renderResult(d) {
  const box = $("results");
  box.hidden = false;

  $("verdict").dataset.tone = TONE[d.verdict_badge] || "safe";
  $("v-title").textContent = d.unified_verdict;
  $("v-statement").textContent = d.summary_statement;
  $("v-score").textContent = d.hybrid_risk_score;
  $("v-conf").textContent = pct(d.confidence, 0);
  $("v-agree").textContent = d.agreement_level;
  $("agree-desc").textContent = d.agreement_description;
  requestAnimationFrame(() => { $("v-marker").style.left = Math.min(100, Math.max(0, d.hybrid_risk_score)) + "%"; });

  renderAnatomy(d.breakdown);
  renderEvidence(d);
  renderDetectors(d.models_comparison);
  renderDrivers(d.models_comparison.machine_learning);
  renderPage(d.optional_page_analysis);
  renderFacts(d.breakdown.anatomy);
  $("disclaimer").textContent = d.disclaimer;

  box.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

function seg(text, cls, label) {
  return el("span", { class: "seg " + cls, text, "data-label": label || "" });
}

function renderAnatomy(b) {
  const a = b.anatomy;
  const line = clear($("anat-line"));
  const isIP = b.has_ip_address;
  const brands = (b.brand_evidence && b.brand_evidence.host) || [];
  const subText = NONE(a.subdomains) ? "" : a.subdomains;
  const subBad = subText && (brands.length > 0 || b.matched_keywords.some((k) => subText.includes(k)));

  if (!NONE(a.scheme)) line.append(seg(a.scheme + "://", "dim", "protocol"));
  if (!NONE(a.userinfo)) line.append(seg(a.userinfo + "@", "bad", "ignored by browser"));
  if (subText && !isIP) line.append(seg(subText + ".", subBad ? "bad" : "dim", subBad ? "disguise" : "subdomain"));
  line.append(seg(isIP ? a.hostname : a.registered_domain || a.hostname, "real", "you are visiting this"));
  if (!NONE(a.port)) line.append(seg(":" + a.port, "dim", "port"));
  const rest = (a.path && a.path !== "/" ? a.path : "") + (NONE(a.query) ? "" : "?" + a.query);
  if (rest) line.append(seg(rest, "path", "path and query"));

  const note = clear($("anat-note"));
  if (!NONE(a.userinfo)) {
    note.append("Everything before the '@' is thrown away by the browser. This link really goes to ", el("strong", { text: a.registered_domain || a.hostname }), ".");
  } else if (subBad) {
    note.append("The start of the address looks familiar on purpose. The site actually belongs to ", el("strong", { text: a.registered_domain }), ".");
  } else if (isIP) {
    note.append("This link points to a raw IP address, so there is no company name to check it against.");
  } else if (b.official_brand) {
    note.append("This is an official ", el("strong", { text: b.official_brand }), " domain.");
  } else {
    note.append("This link goes to ", el("strong", { text: a.registered_domain || a.hostname }), ". If you did not expect that name, stop here.");
  }
}

function evidenceRow(s) {
  const body = el("div", null,
    el("h4", null, s.title, el("span", { class: "by", text: s.source })),
    el("p", { text: s.detail }));
  if (s.why) body.append(el("p", { class: "why", text: s.why }));
  return el("div", { class: "ev" }, el("span", { class: "sev sev-" + s.severity, text: s.severity.charAt(0) + s.severity.slice(1).toLowerCase() }), body);
}

function renderEvidence(d) {
  const wrap = clear($("evidence"));
  const sigs = d.signals_summary || [];
  $("ev-sub").textContent = sigs.length
    ? "Strongest signals first. Rule findings explain themselves; model findings show what the learned patterns reacted to."
    : "";
  if (!sigs.length) {
    wrap.append(el("p", { class: "empty", text: "No warning signs found in the address. That is encouraging, but it does not prove the page is safe." }));
    return;
  }
  const box = el("div", { class: "evidence" });
  sigs.forEach((s) => box.append(evidenceRow(s)));
  wrap.append(box);
}

function renderDetectors(m) {
  const rb = m.rule_based, ml = m.machine_learning;
  $("rb-score").textContent = rb.risk_score;
  $("rb-bar").style.width = rb.risk_score + "%";
  $("rb-what").textContent = rb.verdict;
  $("rb-meta").textContent = `${rb.triggered_rules_count} of ${rb.total_rules_evaluated} rules matched.`;
  $("ml-score").textContent = ml.risk_score;
  $("ml-bar").style.width = ml.risk_score + "%";
  $("ml-what").textContent = ml.verdict;
  const sm = ml.submodels || {};
  $("ml-meta").textContent = `${pct(ml.phishing_probability)} chance of phishing. Measurements model ${pct(sm.feature_model ?? 0, 0)}, character-pattern model ${pct(sm.character_pattern_model ?? 0, 0)}.`;
}

function renderDrivers(ml) {
  const list = clear($("drivers"));
  const top = (ml.top_contributing_features || []).slice(0, 6);
  $("drivers-block").hidden = !top.length;
  const max = Math.max(0.01, ...top.map((f) => Math.abs(f.impact)));
  top.forEach((f) => {
    const w = Math.min(50, (Math.abs(f.impact) / max) * 50);
    const bar = el("i", { class: f.impact >= 0 ? "up" : "down" });
    bar.style.width = w + "%";
    list.append(el("li", null,
      el("span", { class: "lab", text: f.label }),
      el("div", { class: "push", role: "img", ariaLabel: `${f.label}: ${f.impact >= 0 ? "raises" : "lowers"} risk` }, bar),
      el("span", { class: "desc", text: f.description })));
  });
}

function renderPage(p) {
  const blk = $("page-block");
  if (!p) { blk.hidden = true; return; }
  blk.hidden = false;
  const facts = clear($("page-facts"));
  const fnd = clear($("page-findings"));
  if (p.status !== "success") {
    fnd.append(el("p", { class: "empty", text: p.message || "Live inspection was not possible." }));
    return;
  }
  [["Status", String(p.http_status)], ["Title", p.page_title], ["Redirects", `${p.redirect_hops} hop(s)`],
   ["Final address", p.final_url || ""], ["Password field", p.has_password_input ? "Yes" : "No"]]
    .forEach(([k, v]) => { facts.append(el("dt", { text: k }), el("dd", { text: v })); });
  if (p.findings && p.findings.length) {
    const box = el("div", { class: "evidence" });
    p.findings.forEach((f) => box.append(evidenceRow({ severity: f.severity, title: f.title, detail: f.detail, source: "Page" })));
    fnd.append(box);
  } else {
    fnd.append(el("p", { class: "empty", text: "Nothing unusual found on the page itself." }));
  }
}

function renderFacts(a) {
  const dl = clear($("anat-facts"));
  [["Protocol", a.scheme], ["Hostname", a.hostname], ["Subdomains", a.subdomains], ["Registered domain", a.registered_domain],
   ["Extension", a.tld], ["Port", a.port], ["Path", a.path], ["Query", a.query]]
    .forEach(([k, v]) => dl.append(el("dt", { text: k }), el("dd", { text: v })));
}

/* ---------- Evaluation ---------- */
let evalData = null;
let evalSplit = "test";

async function loadEvaluation() {
  if (evalData) return;
  try {
    const res = await fetch("/api/evaluation");
    if (!res.ok) throw new Error();
    evalData = await res.json();
    renderEval();
  } catch (_) {
    clear($("takeaways")).append(el("li", { text: "Could not load the benchmark. Run python evaluate_cli.py to regenerate it." }));
  }
}

function initEvalControls() {
  document.querySelectorAll(".seg-ctl button").forEach((b) =>
    b.addEventListener("click", () => {
      evalSplit = b.dataset.split;
      document.querySelectorAll(".seg-ctl button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderEval();
    })
  );
}

const DETS = [["rule_based_baseline", "Rule engine"], ["machine_learning", "Machine-learning model"], ["hybrid_ensemble", "Hybrid (what you see in the app)"]];

function range(ci, key) {
  const v = ci && ci[key];
  return v && v.length === 2 ? `${pct(v[0], 1)} to ${pct(v[1], 1)}` : "";
}

function renderEval() {
  if (!evalData) return;
  const s = evalData[evalSplit] || evalData;
  const dets = s.detectors;
  const ds = s.dataset || evalData.evaluation_dataset || {};

  $("split-desc").textContent = `${ds.name || ""}: ${ds.total ?? ds.total_test_urls} links (${ds.phishing_count} phishing, ${ds.legitimate_count} legitimate).`;

  const tk = clear($("takeaways"));
  (evalData.key_takeaways || []).forEach((t) => tk.append(el("li", { text: t })));

  const best = (k) => Math.max(...DETS.map(([id]) => dets[id].metrics[k]));
  const bestFpr = Math.min(...DETS.map(([id]) => dets[id].metrics.false_positive_rate));
  const body = clear($("metrics-body"));
  DETS.forEach(([id, name]) => {
    const d = dets[id], m = d.metrics;
    const cell = (v, isBest, ciKey) => {
      const td = el("td", { class: isBest ? "best" : "" }, pct(v));
      const r = range(d.confidence_intervals_95, ciKey);
      if (r) td.append(el("div", { class: "ci", text: r }));
      return td;
    };
    body.append(el("tr", null,
      el("td", null, el("strong", { text: name }), el("small", { text: d.type })),
      cell(m.f1_score, m.f1_score === best("f1_score"), "f1_score"),
      cell(m.precision, m.precision === best("precision")),
      cell(m.recall, m.recall === best("recall"), "recall"),
      cell(m.false_positive_rate, m.false_positive_rate === bestFpr, "false_positive_rate"),
      cell(m.roc_auc, m.roc_auc === best("roc_auc"))));
  });

  const cms = clear($("cms"));
  DETS.forEach(([id, name]) => {
    const c = dets[id].confusion_matrix;
    const cell = (n, label, cls) => el("div", { class: "cm-cell " + cls }, el("b", { text: n }), el("span", { text: label }));
    cms.append(el("div", { class: "cm" }, el("h4", { text: name }),
      el("div", { class: "cm-grid" },
        cell(c.true_positives, "phishing caught", "good"), cell(c.false_negatives, "phishing missed", "bad"),
        cell(c.false_positives, "safe wrongly flagged", "bad"), cell(c.true_negatives, "safe passed", "good"))));
  });

  const src = clear($("by-source"));
  const rows = Object.entries(s.by_source || evalData.by_source || {});
  const scaleMax = Math.max(0.1, ...rows.map(([, v]) => v.hybrid_error_rate)); // fixed 0-10% scale so bars stay honest
  rows.forEach(([name, v]) => {
    const bar = el("i"); bar.style.width = (v.hybrid_error_rate / scaleMax) * 100 + "%";
    src.append(el("div", { class: "src-row" },
      el("div", { class: "nm" }, name, el("small", { text: `${v.count} ${v.label.toLowerCase()} links` })),
      el("div", { class: "barcell" }, el("div", { class: "bar" }, bar)),
      el("div", { class: "pct", text: pct(v.hybrid_error_rate) })));
  });

  const errList = (box, items) => {
    clear(box);
    if (!items || !items.length) { box.append(el("p", { class: "empty", text: "None in this test set." })); return; }
    items.slice(0, 5).forEach((c) => box.append(el("div", { class: "err" },
      el("div", { class: "u", text: c.url }),
      el("div", { class: "t", text: `${c.predicted_verdict} - ${c.source}` }),
      el("div", { class: "r", text: c.root_cause_analysis }))));
  };
  errList($("fps"), dets.hybrid_ensemble.error_cases.false_positives);
  errList($("fns"), dets.hybrid_ensemble.error_cases.false_negatives);

  const lim = clear($("limits"));
  (evalData.limitations || []).forEach((t) => lim.append(el("li", { text: t })));
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initForm();
  initEvalControls();
});
