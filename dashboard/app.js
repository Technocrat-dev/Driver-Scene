/**
 * Driving Scene Generator — Dashboard Application
 *
 * Loads comparison_table.json and evaluation_results JSON files,
 * renders charts and tables for prompt comparison visualization.
 */

// ── Chart Palette ────────────────────────────────────────────────
const COLORS = [
    '#63b3ed', '#68d391', '#f6e05e', '#fc8181',
    '#b794f4', '#f687b3', '#4fd1c5', '#fbd38d'
];

const COLORS_ALPHA = COLORS.map(c => c + '33');

// ── Chart.js Global Defaults ─────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(42, 53, 80, 0.5)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 16;

// ── State ────────────────────────────────────────────────────────
let comparisonData = null;
let evaluationData = {};
let agentData = {};
let charts = {};

// ── DOM Elements ─────────────────────────────────────────────────
const fileInput = document.getElementById('fileInput');
const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('statusText');
const emptyState = document.getElementById('emptyState');

// ── File Loading ─────────────────────────────────────────────────
fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    for (const file of files) {
        try {
            const text = await file.text();
            const data = JSON.parse(text);

            if (file.name.includes('comparison_table')) {
                comparisonData = data;
                setStatus('Loaded', true);
            } else if (file.name.includes('evaluation_results')) {
                const promptId = extractPromptId(file.name);
                evaluationData[promptId] = data;
            } else if (file.name.includes('analysis')) {
                const promptId = extractPromptId(file.name);
                agentData[promptId] = data;
            }
        } catch (err) {
            console.error(`Failed to load ${file.name}:`, err);
        }
    }

    if (comparisonData) {
        renderAll();
    }
});

function extractPromptId(filename) {
    const match = filename.match(/(?:results|analysis)_(.+)\.json/);
    return match ? match[1] : filename;
}

function setStatus(text, active) {
    statusText.textContent = text;
    statusDot.classList.toggle('active', active);
}

// ── Render All ───────────────────────────────────────────────────
function renderAll() {
    emptyState.style.display = 'none';
    renderSummaryCards();
    renderBarChart();
    renderRadarChart();
    renderAccuracyChart();
    renderHallucinationChart();
    renderTable();
    renderAgentAnalysis();
}

// ── Summary Cards ────────────────────────────────────────────────
function renderSummaryCards() {
    const rows = comparisonData;
    const n = rows.length;

    document.getElementById('metricPrompts').textContent = n;
    document.getElementById('metricImages').textContent = rows[0]?.n_images || '—';

    // Find best prompt by composite score
    let bestRow = rows[0];
    let bestScore = -Infinity;
    for (const row of rows) {
        const score = (row.avg_bert_f1 || 0) * 0.3
            + (1 - (row.avg_hallucination_rate || 0)) * 0.3
            + (row.avg_completeness || 0) * 0.25
            + ((row.weather_accuracy || 0) + (row.lighting_accuracy || 0)) / 2 * 0.15;
        if (score > bestScore) {
            bestScore = score;
            bestRow = row;
        }
    }

    document.getElementById('metricBestPrompt').textContent = bestRow.prompt_id;
    document.getElementById('metricBestScore').textContent =
        `Composite: ${(bestScore * 100).toFixed(1)}%`;

    const avgHall = rows.reduce((s, r) => s + (r.avg_hallucination_rate || 0), 0) / n;
    document.getElementById('metricHallucination').textContent =
        (avgHall * 100).toFixed(1) + '%';
}

// ── Bar Chart (Key Metrics) ──────────────────────────────────────
function renderBarChart() {
    if (charts.bar) charts.bar.destroy();

    const labels = comparisonData.map(r => r.prompt_id.replace('v', 'V').replace('_', ' '));
    const bertScores = comparisonData.map(r => r.avg_bert_f1 || 0);
    const completeness = comparisonData.map(r => r.avg_completeness || 0);
    const invHall = comparisonData.map(r => 1 - (r.avg_hallucination_rate || 0));

    charts.bar = new Chart(document.getElementById('barChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'BERTScore F1',
                    data: bertScores,
                    backgroundColor: COLORS[0] + 'cc',
                    borderColor: COLORS[0],
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'Completeness',
                    data: completeness,
                    backgroundColor: COLORS[1] + 'cc',
                    borderColor: COLORS[1],
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: '1 - Hallucination',
                    data: invHall,
                    backgroundColor: COLORS[2] + 'cc',
                    borderColor: COLORS[2],
                    borderWidth: 1,
                    borderRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.05,
                    ticks: { callback: v => (v * 100).toFixed(0) + '%' },
                    grid: { color: 'rgba(42, 53, 80, 0.3)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 }, maxRotation: 45 }
                }
            }
        }
    });
}

// ── Radar Chart (Top 3) ──────────────────────────────────────────
function renderRadarChart() {
    if (charts.radar) charts.radar.destroy();

    // Pick top 3 by composite score
    const scored = comparisonData.map(r => ({
        ...r,
        composite: (r.avg_bert_f1 || 0) * 0.3
            + (1 - (r.avg_hallucination_rate || 0)) * 0.3
            + (r.avg_completeness || 0) * 0.25
            + ((r.weather_accuracy || 0) + (r.lighting_accuracy || 0)) / 2 * 0.15
    }));
    scored.sort((a, b) => b.composite - a.composite);
    const top3 = scored.slice(0, 3);

    const radarLabels = ['BERTScore', 'Completeness', '1-Halluc.', 'Weather Acc', 'Lighting Acc'];

    charts.radar = new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: radarLabels,
            datasets: top3.map((r, i) => ({
                label: r.prompt_id,
                data: [
                    r.avg_bert_f1 || 0,
                    r.avg_completeness || 0,
                    1 - (r.avg_hallucination_rate || 0),
                    r.weather_accuracy || 0,
                    r.lighting_accuracy || 0
                ],
                borderColor: COLORS[i],
                backgroundColor: COLORS_ALPHA[i],
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: COLORS[i],
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 11 } } }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(42, 53, 80, 0.4)' },
                    grid: { color: 'rgba(42, 53, 80, 0.3)' },
                    pointLabels: { font: { size: 10 } },
                    suggestedMin: 0,
                    suggestedMax: 1,
                    ticks: {
                        display: false,
                        stepSize: 0.25
                    }
                }
            }
        }
    });
}

// ── Accuracy Chart ───────────────────────────────────────────────
function renderAccuracyChart() {
    if (charts.accuracy) charts.accuracy.destroy();

    const labels = comparisonData.map(r => r.prompt_id.replace('v', 'V').replace('_', ' '));
    const weather = comparisonData.map(r => (r.weather_accuracy || 0) * 100);
    const lighting = comparisonData.map(r => (r.lighting_accuracy || 0) * 100);

    charts.accuracy = new Chart(document.getElementById('accuracyChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Weather Accuracy %',
                    data: weather,
                    backgroundColor: COLORS[4] + 'cc',
                    borderColor: COLORS[4],
                    borderWidth: 1,
                    borderRadius: 4,
                },
                {
                    label: 'Lighting Accuracy %',
                    data: lighting,
                    backgroundColor: COLORS[6] + 'cc',
                    borderColor: COLORS[6],
                    borderWidth: 1,
                    borderRadius: 4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 105,
                    ticks: { callback: v => v + '%' },
                    grid: { color: 'rgba(42, 53, 80, 0.3)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 }, maxRotation: 45 }
                }
            }
        }
    });
}

// ── Hallucination Chart ──────────────────────────────────────────
function renderHallucinationChart() {
    if (charts.hall) charts.hall.destroy();

    const labels = comparisonData.map(r => r.prompt_id.replace('v', 'V').replace('_', ' '));
    const rates = comparisonData.map(r => (r.avg_hallucination_rate || 0) * 100);

    const barColors = rates.map(r =>
        r <= 10 ? COLORS[1] + 'cc' :
            r <= 25 ? COLORS[2] + 'cc' :
                COLORS[3] + 'cc'
    );

    charts.hall = new Chart(document.getElementById('hallucinationChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Hallucination Rate %',
                data: rates,
                backgroundColor: barColors,
                borderWidth: 0,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Hallucination: ${ctx.raw.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { callback: v => v + '%' },
                    grid: { color: 'rgba(42, 53, 80, 0.3)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 10 }, maxRotation: 45 }
                }
            }
        }
    });
}

// ── Comparison Table ─────────────────────────────────────────────
function renderTable() {
    const tbody = document.getElementById('tableBody');

    // Find best composite row
    let bestIdx = 0;
    let bestScore = -Infinity;
    comparisonData.forEach((r, i) => {
        const score = (r.avg_bert_f1 || 0) * 0.3
            + (1 - (r.avg_hallucination_rate || 0)) * 0.3
            + (r.avg_completeness || 0) * 0.25
            + ((r.weather_accuracy || 0) + (r.lighting_accuracy || 0)) / 2 * 0.15;
        if (score > bestScore) { bestScore = score; bestIdx = i; }
    });

    tbody.innerHTML = comparisonData.map((row, i) => {
        const bert = row.avg_bert_f1 || 0;
        const hall = row.avg_hallucination_rate || 0;
        const comp = row.avg_completeness || 0;

        const bertClass = bert >= 0.3 ? 'cell-good' : bert >= 0.2 ? 'cell-warn' : 'cell-bad';
        const hallClass = hall <= 0.1 ? 'cell-good' : hall <= 0.3 ? 'cell-warn' : 'cell-bad';
        const compClass = comp >= 0.8 ? 'cell-good' : comp >= 0.5 ? 'cell-warn' : 'cell-bad';

        return `<tr class="${i === bestIdx ? 'best-row' : ''}">
            <td>${row.prompt_id}</td>
            <td>${row.prompt_strategy || '—'}</td>
            <td>${row.n_images}</td>
            <td class="${bertClass}">${(bert * 100).toFixed(1)}%</td>
            <td class="${hallClass}">${(hall * 100).toFixed(1)}%</td>
            <td class="${compClass}">${(comp * 100).toFixed(1)}%</td>
            <td>${(row.avg_count_accuracy_mae || 0).toFixed(2)}</td>
            <td>${((row.weather_accuracy || 0) * 100).toFixed(0)}%</td>
            <td>${((row.lighting_accuracy || 0) * 100).toFixed(0)}%</td>
            <td>${row.avg_judge_score != null ? row.avg_judge_score.toFixed(1) : '—'}</td>
            <td>${((row.avg_spatial_accuracy || 0) * 100).toFixed(1)}%</td>
        </tr>`;
    }).join('');
}

// ── Agent Analysis ───────────────────────────────────────────────
function renderAgentAnalysis() {
    const section = document.getElementById('agentSection');
    const content = document.getElementById('agentContent');
    const keys = Object.keys(agentData);

    if (keys.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    let html = '';

    for (const promptId of keys) {
        const report = agentData[promptId];
        html += `<h3 style="color: var(--accent); margin: 1rem 0 0.5rem;">${report.prompt_id || promptId}</h3>`;
        html += `<p style="font-size: 0.82rem; margin-bottom: 0.75rem; color: var(--text-secondary);">${report.overall_assessment || ''}</p>`;

        if (report.strengths && report.strengths.length) {
            html += `<div style="margin-bottom: 0.75rem;"><strong style="color: var(--green); font-size: 0.78rem;">Strengths:</strong>`;
            for (const s of report.strengths) {
                html += `<p style="font-size: 0.78rem; color: var(--text-secondary); padding-left: 1rem;">+ ${s}</p>`;
            }
            html += `</div>`;
        }

        if (report.error_patterns && report.error_patterns.length) {
            for (const p of report.error_patterns) {
                html += `<div class="agent-pattern">
                    <span class="severity severity-${p.severity}">${p.severity}</span>
                    <h4>${p.pattern_type}</h4>
                    <p>${p.description}</p>
                    ${p.suggestion ? `<div class="suggestion-block">${p.suggestion}</div>` : ''}
                </div>`;
            }
        }

        if (report.improvement_suggestions && report.improvement_suggestions.length) {
            html += `<div style="margin-top: 0.75rem;"><strong style="color: var(--accent); font-size: 0.78rem;">Improvement Suggestions:</strong>`;
            for (const s of report.improvement_suggestions) {
                html += `<div class="suggestion-block" style="margin-top: 0.4rem;">${s}</div>`;
            }
            html += `</div>`;
        }
    }

    content.innerHTML = html;
}

// ── Init ─────────────────────────────────────────────────────────
setStatus('No data loaded', false);

// Auto-load sample data if available
(async () => {
    try {
        const response = await fetch('./sample_comparison_table.json');
        if (response.ok) {
            comparisonData = await response.json();
            setStatus('Sample data', true);
            renderAll();
        }
    } catch (_) {
        // Sample file not available — user must load manually
    }
})();

