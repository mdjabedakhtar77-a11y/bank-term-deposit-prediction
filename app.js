// Bank Marketing Campaign Intelligence & ML Engine
let dashboardData = null;
let charts = {};

// Initialize application
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  await loadDashboardData();
  setupFilters();
  simulateScore(); // Initialize simulator
});

// Tab Navigation
function setupNavigation() {
  const navLinks = document.querySelectorAll(".nav-item");
  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const tab = link.getAttribute("data-tab");
      switchTab(tab);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.getAttribute("data-tab") === tabId);
  });
  document.querySelectorAll(".content-section").forEach(sec => {
    sec.classList.toggle("active", sec.id === tabId);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

window.scrollToSection = function(sectionId) {
  switchTab(sectionId);
};

// Fetch json data
async function loadDashboardData() {
  try {
    const res = await fetch("dashboard_data.json");
    dashboardData = await res.json();
    populateKPIs();
    renderAllCharts();
    populateLeadsTable(dashboardData.priority_leads);
  } catch (err) {
    console.error("Failed to load dashboard_data.json:", err);
  }
}

// Populate Top KPIs
function populateKPIs() {
  const ov = dashboardData.overview;
  const rf = dashboardData.models.random_forest;
  const dt = dashboardData.models.decision_tree;

  document.getElementById("kpiTotal").textContent = ov.total_customers.toLocaleString();
  document.getElementById("kpiRate").textContent = `${ov.subscription_rate}%`;
  document.getElementById("kpiSubscribers").textContent = ov.subscribers.toLocaleString();
  document.getElementById("kpiAuc").textContent = rf.roc_auc.toFixed(3);

  // Model evaluation numbers
  document.getElementById("dtAcc").textContent = `${(dt.accuracy * 100).toFixed(2)}%`;
  document.getElementById("dtPrec").textContent = `${(dt.precision * 100).toFixed(2)}%`;
  document.getElementById("dtRec").textContent = `${(dt.recall * 100).toFixed(2)}%`;
  document.getElementById("dtF1").textContent = dt.f1.toFixed(4);
  document.getElementById("dtAuc").textContent = dt.roc_auc.toFixed(4);

  document.getElementById("rfAcc").textContent = `${(rf.accuracy * 100).toFixed(2)}%`;
  document.getElementById("rfPrec").textContent = `${(rf.precision * 100).toFixed(2)}%`;
  document.getElementById("rfRec").textContent = `${(rf.recall * 100).toFixed(2)}%`;
  document.getElementById("rfF1").textContent = rf.f1.toFixed(4);
  document.getElementById("rfAuc").textContent = rf.roc_auc.toFixed(4);

  // Render Confusion Matrices
  const dtCm = dt.confusion_matrix;
  document.getElementById("dtCm").innerHTML = `
    <div class="cm-cell true-neg">TN: ${dtCm.tn}</div>
    <div class="cm-cell false-pos">FP: ${dtCm.fp}</div>
    <div class="cm-cell false-neg">FN: ${dtCm.fn}</div>
    <div class="cm-cell true-pos">TP: ${dtCm.tp}</div>
  `;

  const rfCm = rf.confusion_matrix;
  document.getElementById("rfCm").innerHTML = `
    <div class="cm-cell true-neg">TN: ${rfCm.tn}</div>
    <div class="cm-cell false-pos">FP: ${rfCm.fp}</div>
    <div class="cm-cell false-neg">FN: ${rfCm.fn}</div>
    <div class="cm-cell true-pos">TP: ${rfCm.tp}</div>
  `;
}

// Chart rendering using Chart.js
function renderAllCharts() {
  const ch = dashboardData.charts;
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "rgba(255, 255, 255, 0.08)";
  Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

  // 1. Cumulative Gain Chart
  const ctxGain = document.getElementById("cumulativeGainChart").getContext("2d");
  charts.gain = new Chart(ctxGain, {
    type: "line",
    data: {
      labels: ch.cumulative_gain.pct_contacted.map(v => `${v}%`),
      datasets: [
        {
          label: "Random Forest (Prioritized)",
          data: ch.cumulative_gain.pct_captured,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          borderWidth: 3,
          fill: true,
          tension: 0.3
        },
        {
          label: "Random Dialing (No Model Baseline)",
          data: ch.cumulative_gain.baseline,
          borderColor: "#64748b",
          borderDash: [5, 5],
          borderWidth: 2,
          fill: false,
          tension: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}% subscribers captured`
          }
        }
      },
      scales: {
        y: { min: 0, max: 100, title: { display: true, text: "% of All Subscribers Captured" } },
        x: { title: { display: true, text: "% of Total Client Base Contacted" } }
      }
    }
  });

  // 2. Feature Importance Chart
  const ctxFeat = document.getElementById("featureImportanceChart").getContext("2d");
  charts.feat = new Chart(ctxFeat, {
    type: "bar",
    data: {
      labels: ch.feature_importance.features,
      datasets: [{
        label: "Feature Importance Score",
        data: ch.feature_importance.scores,
        backgroundColor: "#3b82f6",
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, title: { display: true, text: "Gini Importance" } }
      }
    }
  });

  // 3. Job Conversion Rate Chart
  const ctxJob = document.getElementById("jobChart").getContext("2d");
  charts.job = new Chart(ctxJob, {
    type: "bar",
    data: {
      labels: ch.job_chart.categories,
      datasets: [{
        label: "Conversion Rate (%)",
        data: ch.job_chart.conversion_rates,
        backgroundColor: ch.job_chart.conversion_rates.map(r => r > 15 ? "#10b981" : "#3b82f6"),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Conversion: ${ctx.raw}%`
          }
        }
      },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "% Subscribed" } }
      }
    }
  });

  // 4. Age Demographic Stacked Chart
  const ctxAge = document.getElementById("ageChart").getContext("2d");
  charts.age = new Chart(ctxAge, {
    type: "bar",
    data: {
      labels: ch.age_chart.labels,
      datasets: [
        {
          label: "Subscribed (Yes)",
          data: ch.age_chart.yes,
          backgroundColor: "#10b981",
          borderRadius: 4
        },
        {
          label: "Did Not Subscribe (No)",
          data: ch.age_chart.no,
          backgroundColor: "rgba(255, 255, 255, 0.12)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true }
      }
    }
  });

  // 5. Education Chart
  const ctxEdu = document.getElementById("educationChart").getContext("2d");
  charts.edu = new Chart(ctxEdu, {
    type: "doughnut",
    data: {
      labels: ch.education_chart.categories,
      datasets: [{
        data: ch.education_chart.totals,
        backgroundColor: ["#3b82f6", "#8b5cf6", "#10b981", "#64748b"]
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "right" } }
    }
  });

  // 6. Marital Status Chart
  const ctxMar = document.getElementById("maritalChart").getContext("2d");
  charts.mar = new Chart(ctxMar, {
    type: "bar",
    data: {
      labels: ch.marital_chart.categories,
      datasets: [{
        label: "Conversion %",
        data: ch.marital_chart.conversion_rates,
        backgroundColor: "#8b5cf6",
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });

  // 7. Month Chart
  const ctxMon = document.getElementById("monthChart").getContext("2d");
  charts.mon = new Chart(ctxMon, {
    type: "line",
    data: {
      labels: ch.month_chart.categories,
      datasets: [{
        label: "Conversion %",
        data: ch.month_chart.conversion_rates,
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.2)",
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });

  // 8. ROC Curves Comparison
  const ctxRoc = document.getElementById("rocChart").getContext("2d");
  const dtRoc = dashboardData.models.decision_tree.roc_curve;
  const rfRoc = dashboardData.models.random_forest.roc_curve;

  charts.roc = new Chart(ctxRoc, {
    type: "line",
    data: {
      datasets: [
        {
          label: `Random Forest (AUC = ${dashboardData.models.random_forest.roc_auc})`,
          data: rfRoc.fpr.map((x, i) => ({ x: x, y: rfRoc.tpr[i] })),
          borderColor: "#10b981",
          borderWidth: 2.5,
          tension: 0.2
        },
        {
          label: `Decision Tree (AUC = ${dashboardData.models.decision_tree.roc_auc})`,
          data: dtRoc.fpr.map((x, i) => ({ x: x, y: dtRoc.tpr[i] })),
          borderColor: "#f59e0b",
          borderWidth: 2,
          tension: 0.2
        },
        {
          label: "Random Guess (AUC = 0.500)",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: "#64748b",
          borderDash: [5, 5],
          borderWidth: 1.5
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False Positive Rate (1 - Specificity)" } },
        y: { type: "linear", min: 0, max: 1, title: { display: true, text: "True Positive Rate (Recall / Sensitivity)" } }
      }
    }
  });
}

// Populate Priority Leads Queue Table
function populateLeadsTable(leads) {
  const tbody = document.getElementById("leadsTableBody");
  tbody.innerHTML = "";

  leads.forEach(lead => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="rank-badge">#${lead.rank}</span></td>
      <td>
        <div class="prob-meter-wrap">
          <div class="prob-meter-bar">
            <div class="prob-fill" style="width: ${lead.probability}%"></div>
          </div>
          <span class="prob-pct">${lead.probability}%</span>
        </div>
      </td>
      <td><strong>${lead.age}</strong> yrs</td>
      <td><span style="text-transform: capitalize;">${lead.job}</span></td>
      <td><span style="text-transform: capitalize;">${lead.education}</span></td>
      <td>&#8377;${lead.balance.toLocaleString('en-IN')}</td>
      <td>${lead.housing === 'yes' ? 'Yes' : 'No'}</td>
      <td>${lead.duration}s (${lead.campaign} call)</td>
      <td>
        <span class="${lead.actual === 'yes' ? 'badge-deposit-yes' : 'badge-deposit-no'}">
          ${lead.actual.toUpperCase()}
        </span>
      </td>
      <td>
        <button class="btn-dial" onclick="alert('Calling Lead #${lead.rank} (Est. Success: ${lead.probability}%)')">
          Dial Client
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Filter and search queue
function setupFilters() {
  const searchInput = document.getElementById("leadSearch");
  const probFilter = document.getElementById("probFilter");

  function applyFilter() {
    if (!dashboardData) return;
    const term = searchInput.value.toLowerCase();
    const minProb = probFilter.value === "all" ? 0 : parseFloat(probFilter.value);

    const filtered = dashboardData.priority_leads.filter(lead => {
      const matchText = lead.job.toLowerCase().includes(term) ||
                        lead.education.toLowerCase().includes(term) ||
                        lead.marital.toLowerCase().includes(term);
      const matchProb = lead.probability >= minProb;
      return matchText && matchProb;
    });

    populateLeadsTable(filtered);
  }

  searchInput.addEventListener("input", applyFilter);
  probFilter.addEventListener("change", applyFilter);
}

// Interactive Propensity Simulator
window.simulateScore = function() {
  const age = parseFloat(document.getElementById("simAge").value) || 35;
  const job = document.getElementById("simJob").value;
  const balance = parseFloat(document.getElementById("simBalance").value) || 0;
  const duration = parseFloat(document.getElementById("simDuration").value) || 300;
  const campaign = parseFloat(document.getElementById("simCampaign").value) || 1;
  const housing = document.getElementById("simHousing").value;
  const loan = document.getElementById("simLoan").value;
  const poutcome = document.getElementById("simPoutcome").value;

  // Approximate propensity score model based on trained tree weights
  let logit = -1.6;

  // Duration impact (single most dominant feature)
  logit += Math.min(duration / 320, 2.5) * 1.4;

  // Previous success
  if (poutcome === "success") logit += 1.8;
  else if (poutcome === "failure") logit -= 0.3;

  // Job factors
  if (job === "student" || job === "retired") logit += 0.85;
  else if (job === "management") logit += 0.2;
  else if (job === "blue-collar") logit -= 0.4;

  // Balance (in ₹ INR)
  if (balance > 50000) logit += 0.5;
  else if (balance > 15000) logit += 0.25;
  else if (balance < 0) logit -= 0.6;

  // Age (young or senior)
  if (age < 25 || age > 60) logit += 0.45;

  // Debt penalty
  if (housing === "yes") logit -= 0.4;
  if (loan === "yes") logit -= 0.35;

  // Fatigue penalty
  if (campaign > 3) logit -= (campaign - 3) * 0.2;

  // Sigmoid conversion to probability percentage
  const prob = 1 / (1 + Math.exp(-logit));
  const pct = Math.min(Math.max(Math.round(prob * 100), 4), 98);

  const circle = document.getElementById("scoreCircle");
  const numElem = document.getElementById("scoreValue");
  const badge = document.getElementById("decisionBadge");
  const insight = document.getElementById("simInsights");

  numElem.textContent = `${pct}%`;

  if (pct >= 65) {
    circle.style.borderColor = "#10b981";
    circle.style.boxShadow = "0 0 32px rgba(16, 185, 129, 0.3)";
    badge.className = "decision-badge";
    badge.style.color = "#10b981";
    badge.style.borderColor = "rgba(16, 185, 129, 0.3)";
    badge.style.background = "rgba(16, 185, 129, 0.15)";
    badge.textContent = "HIGH PRIORITY LEAD — DIAL IMMEDIATELY";
    insight.innerHTML = "Strong indicators: Receptive job group, low debt exposure, positive past track record.";
  } else if (pct >= 35) {
    circle.style.borderColor = "#f59e0b";
    circle.style.boxShadow = "0 0 32px rgba(245, 158, 11, 0.3)";
    badge.className = "decision-badge";
    badge.style.color = "#f59e0b";
    badge.style.borderColor = "rgba(245, 158, 11, 0.3)";
    badge.style.background = "rgba(245, 158, 11, 0.15)";
    badge.textContent = "MODERATE PRIORITY — SECONDARY CAMPAIGN";
    insight.innerHTML = "Average probability: Consider digital nurture or SMS follow-up before initiating direct outbound phone calls.";
  } else {
    circle.style.borderColor = "#ef4444";
    circle.style.boxShadow = "0 0 32px rgba(239, 68, 68, 0.3)";
    badge.className = "decision-badge";
    badge.style.color = "#ef4444";
    badge.style.borderColor = "rgba(239, 68, 68, 0.3)";
    badge.style.background = "rgba(239, 68, 68, 0.15)";
    badge.textContent = "LOW PRIORITY — DO NOT CALL";
    insight.innerHTML = "Very low conversion chance: Customer possesses debt burden or fatigue from frequent previous contact attempts.";
  }
};
