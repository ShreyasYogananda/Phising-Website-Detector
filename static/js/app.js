/**
 * Explainable Phishing Risk Checker - Front-End Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSampleChips();
  initAnalyzeForm();
  loadEvaluationDashboard();
});

// 1. Navigation Tabs
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab-btn");
  const sections = document.querySelectorAll(".tab-content");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.target;
      tabs.forEach(t => t.classList.remove("active"));
      sections.forEach(s => s.style.display = "none");

      tab.classList.add("active");
      const activeSection = document.getElementById(target);
      if (activeSection) {
        activeSection.style.display = "block";
      }

      if (target === "tab-eval") {
        loadEvaluationDashboard();
      }
    });
  });
}

// 2. Sample URL Chips
function initSampleChips() {
  const chips = document.querySelectorAll(".sample-chip");
  const input = document.getElementById("url-input");

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const url = chip.dataset.url;
      if (url) {
        input.value = url;
        input.focus();
      }
    });
  });
}

// 3. Form Submission & Analysis
function initAnalyzeForm() {
  const form = document.getElementById("analyze-form");
  const input = document.getElementById("url-input");
  const btn = document.getElementById("analyze-btn");
  const pageCheck = document.getElementById("page-analysis-checkbox");
  const resultsDiv = document.getElementById("results-area");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = input.value.trim();
    if (!url) return;

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Analyzing...`;
    resultsDiv.style.display = "none";

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: url,
          enable_page_analysis: pageCheck ? pageCheck.checked : false
        })
      });

      if (!response.ok) {
        const err = await response.json();
        alert("Error: " + (err.error || "Analysis failed"));
        return;
      }

      const data = await response.json();
      renderAnalysisResults(data);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze URL: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `Analyze Risk`;
    }
  });
}

// 4. Render Risk Results
function renderAnalysisResults(data) {
  const resultsDiv = document.getElementById("results-area");
  resultsDiv.style.display = "block";
  resultsDiv.scrollIntoView({ behavior: "smooth", block: "start" });

  const score = data.hybrid_risk_score;
  const verdict = data.unified_verdict;
  const badgeClass = data.verdict_badge; // danger, warning, success

  // Verdict Banner
  const banner = document.getElementById("verdict-banner");
  banner.className = `verdict-banner ${badgeClass}`;

  const badgeElem = document.getElementById("verdict-badge");
  badgeElem.className = `verdict-badge ${badgeClass}`;
  badgeElem.textContent = verdict;

  document.getElementById("summary-statement").textContent = data.summary_statement;
  document.getElementById("gauge-score").textContent = score;

  // Confidence & Uncertainty
  const confPct = Math.round(data.confidence * 100);
  const uncertPct = Math.round(data.uncertainty * 100);
  document.getElementById("meta-confidence").textContent = `${confPct}%`;
  document.getElementById("meta-uncertainty").textContent = `${uncertPct}%`;
  document.getElementById("meta-agreement").textContent = data.agreement_level;

  // SVG Gauge Needle Rotation
  // 0 -> -90 deg, 100 -> 90 deg
  const deg = -90 + (score / 100) * 180;
  const needle = document.getElementById("gauge-needle");
  if (needle) {
    needle.setAttribute("transform", `rotate(${deg} 90 90)`);
  }

  // Detectors Comparison Cards
  const rb = data.models_comparison.rule_based;
  const ml = data.models_comparison.machine_learning;

  document.getElementById("rb-score").textContent = `${rb.risk_score} / 100`;
  document.getElementById("rb-verdict").textContent = rb.verdict;
  document.getElementById("rb-rules-count").textContent = `${rb.triggered_rules_count} of ${rb.total_rules_evaluated} rules`;

  document.getElementById("ml-score").textContent = `${ml.risk_score} / 100`;
  document.getElementById("ml-verdict").textContent = ml.verdict;
  document.getElementById("ml-prob").textContent = `${(ml.phishing_probability * 100).toFixed(1)}%`;
  document.getElementById("ml-uncertainty").textContent = `${Math.round(ml.uncertainty * 100)}%`;

  // Signals List
  const signalsContainer = document.getElementById("signals-list");
  signalsContainer.innerHTML = "";

  data.signals_summary.forEach(sig => {
    const row = document.createElement("div");
    row.className = "signal-row";
    row.innerHTML = `
      <span class="severity-pill severity-${sig.severity}">${sig.severity}</span>
      <div class="signal-content">
        <div class="signal-title">${escapeHtml(sig.title)} <small style="color:var(--text-dim);">[${sig.source}]</small></div>
        <div class="signal-detail">${escapeHtml(sig.detail)}</div>
      </div>
    `;
    signalsContainer.appendChild(row);
  });

  // URL Anatomy
  const anat = data.breakdown.anatomy;
  document.getElementById("anat-scheme").textContent = anat.scheme;
  document.getElementById("anat-host").textContent = anat.hostname;
  document.getElementById("anat-sub").textContent = anat.subdomains;
  document.getElementById("anat-reg").textContent = anat.registered_domain;
  document.getElementById("anat-tld").textContent = anat.tld;
  document.getElementById("anat-path").textContent = anat.path;
  document.getElementById("anat-query").textContent = anat.query;

  // Optional Page Analysis
  const pageContainer = document.getElementById("page-analysis-section");
  if (data.optional_page_analysis) {
    pageContainer.style.display = "block";
    const p = data.optional_page_analysis;
    document.getElementById("page-http-status").textContent = p.http_status || "N/A";
    document.getElementById("page-title-val").textContent = p.page_title || "(none)";
    document.getElementById("page-redirects-val").textContent = `${p.redirect_hops} hop(s)`;
    document.getElementById("page-pass-val").textContent = p.has_password_input ? "Yes (Detected)" : "No";

    const pFindings = document.getElementById("page-findings-list");
    pFindings.innerHTML = "";
    if (p.findings && p.findings.length > 0) {
      p.findings.forEach(f => {
        const d = document.createElement("div");
        d.className = "signal-row";
        d.innerHTML = `
          <span class="severity-pill severity-${f.severity}">${f.severity}</span>
          <div class="signal-content">
            <div class="signal-title">${escapeHtml(f.title)}</div>
            <div class="signal-detail">${escapeHtml(f.detail)}</div>
          </div>
        `;
        pFindings.appendChild(d);
      });
    } else {
      pFindings.innerHTML = `<div style="color:var(--text-muted); font-size:0.85rem;">No anomalous page behavior detected during isolated inspection.</div>`;
    }
  } else {
    pageContainer.style.display = "none";
  }

  // Security Disclaimer
  document.getElementById("disclaimer-text").textContent = data.disclaimer;
}

// 5. Load Evaluation Dashboard
async function loadEvaluationDashboard() {
  try {
    const res = await fetch("/api/evaluation");
    if (!res.ok) return;
    const data = await res.json();
    renderEvaluationDashboard(data);
  } catch (err) {
    console.error("Failed to load evaluation metrics", err);
  }
}

function renderEvaluationDashboard(data) {
  const d = data.detectors;

  // Comparison Metrics Table
  const tbody = document.getElementById("eval-table-body");
  if (!tbody) return;

  const rows = [
    {
      name: d.rule_based_baseline.name,
      type: "Heuristic Rules",
      acc: d.rule_based_baseline.metrics.accuracy,
      prec: d.rule_based_baseline.metrics.precision,
      rec: d.rule_based_baseline.metrics.recall,
      f1: d.rule_based_baseline.metrics.f1_score,
      fpr: d.rule_based_baseline.metrics.false_positive_rate
    },
    {
      name: d.machine_learning.name,
      type: "Random Forest ML",
      acc: d.machine_learning.metrics.accuracy,
      prec: d.machine_learning.metrics.precision,
      rec: d.machine_learning.metrics.recall,
      f1: d.machine_learning.metrics.f1_score,
      fpr: d.machine_learning.metrics.false_positive_rate
    },
    {
      name: d.hybrid_ensemble.name,
      type: "Hybrid Ensemble",
      acc: d.hybrid_ensemble.metrics.accuracy,
      prec: d.hybrid_ensemble.metrics.precision,
      rec: d.hybrid_ensemble.metrics.recall,
      f1: d.hybrid_ensemble.metrics.f1_score,
      fpr: d.hybrid_ensemble.metrics.false_positive_rate
    }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${escapeHtml(r.name)}</strong></td>
      <td><span class="meta-pill">${escapeHtml(r.type)}</span></td>
      <td><strong>${(r.f1 * 100).toFixed(1)}%</strong></td>
      <td>${(r.prec * 100).toFixed(1)}%</td>
      <td>${(r.rec * 100).toFixed(1)}%</td>
      <td>${(r.acc * 100).toFixed(1)}%</td>
      <td>${(r.fpr * 100).toFixed(1)}%</td>
    </tr>
  `).join("");

  // Confusion Matrices
  renderConfusionMatrix("rb-cm", d.rule_based_baseline.confusion_matrix);
  renderConfusionMatrix("ml-cm", d.machine_learning.confusion_matrix);

  // Error Cases (False Positives and False Negatives)
  renderErrorCases("rb-fps", d.rule_based_baseline.error_cases.false_positives, "fp");
  renderErrorCases("rb-fns", d.rule_based_baseline.error_cases.false_negatives, "fn");
  renderErrorCases("ml-fps", d.machine_learning.error_cases.false_positives, "fp");
  renderErrorCases("ml-fns", d.machine_learning.error_cases.false_negatives, "fn");
}

function renderConfusionMatrix(containerId, cm) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = `
    <div class="cm-grid">
      <div></div>
      <div style="font-weight:600; color:var(--text-muted);">Pred Legit</div>
      <div style="font-weight:600; color:var(--text-muted);">Pred Phish</div>

      <div style="font-weight:600; color:var(--text-muted); display:flex; align-items:center;">Actual Legit</div>
      <div class="cm-cell tn">
        <span class="cm-val">${cm.true_negatives}</span>
        <span>True Neg (TN)</span>
      </div>
      <div class="cm-cell fp">
        <span class="cm-val">${cm.false_positives}</span>
        <span>False Pos (FP)</span>
      </div>

      <div style="font-weight:600; color:var(--text-muted); display:flex; align-items:center;">Actual Phish</div>
      <div class="cm-cell fn">
        <span class="cm-val">${cm.false_negatives}</span>
        <span>False Neg (FN)</span>
      </div>
      <div class="cm-cell tp">
        <span class="cm-val">${cm.true_positives}</span>
        <span>True Pos (TP)</span>
      </div>
    </div>
  `;
}

function renderErrorCases(containerId, cases, type) {
  const el = document.getElementById(containerId);
  if (!el) return;

  if (!cases || cases.length === 0) {
    el.innerHTML = `<div style="color:var(--text-dim); font-size:0.85rem;">None observed in sample window.</div>`;
    return;
  }

  el.innerHTML = cases.map(c => `
    <div class="error-card ${type}">
      <div class="error-url">${escapeHtml(c.url)}</div>
      <div style="font-size:0.8rem; margin-bottom:4px;">
        <span style="color:var(--text-muted);">True:</span> <strong>${escapeHtml(c.true_label)}</strong> |
        <span style="color:var(--text-muted);">Result:</span> <strong>${escapeHtml(c.predicted_verdict)}</strong>
      </div>
      <div class="error-reason"><strong>Analysis:</strong> ${escapeHtml(c.root_cause_analysis)}</div>
    </div>
  `).join("");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
