/**
 * app.js — UI Controller for Explainable Summary Score
 * Wires the DOM to scorer.js, handles dark/light toggle,
 * renders the explanation & SHAP chart, manages batch CSV scoring.
 */

'use strict';

// ─────────────────────────────────────────
// Theme
// ─────────────────────────────────────────

const html = document.documentElement;
const themeToggle = document.getElementById('theme-toggle');
const THEME_KEY = 'ess-theme';

function applyTheme(t) {
    html.setAttribute('data-theme', t);
    themeToggle.textContent = t === 'dark' ? '☀️' : '🌙';
    localStorage.setItem(THEME_KEY, t);
}

themeToggle.addEventListener('click', () => {
    applyTheme(html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
});

const savedTheme = localStorage.getItem(THEME_KEY) ||
    (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
applyTheme(savedTheme);

// ─────────────────────────────────────────
// Navbar scroll class
// ─────────────────────────────────────────

const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 10);
}, { passive: true });

// ─────────────────────────────────────────
// Scroll-reveal
// ─────────────────────────────────────────

const appearEls = document.querySelectorAll('.appear');
const observer = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('in-view'); });
}, { threshold: 0.1 });
appearEls.forEach(el => observer.observe(el));

// ─────────────────────────────────────────
// Toast
// ─────────────────────────────────────────

function showToast(msg, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = (type === 'success' ? '✅ ' : '❌ ') + msg;
    toast.className = `toast toast--${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3500);
}

// ─────────────────────────────────────────
// Smooth scroll nav links
// ─────────────────────────────────────────

document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        const t = document.querySelector(a.getAttribute('href'));
        if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
});

// ─────────────────────────────────────────
// Score colour helpers
// ─────────────────────────────────────────

function scoreColor(score, maxScore) {
    const p = score / maxScore;
    if (p >= 0.70) return '#4ade80';
    if (p >= 0.40) return '#fbbf24';
    return '#f87171';
}

function scoreColorClass(score, maxScore) {
    const p = score / maxScore;
    if (p >= 0.70) return 'score-chip--high';
    if (p >= 0.40) return 'score-chip--mid';
    return 'score-chip--low';
}

// ─────────────────────────────────────────
// RENDER — Score Display (two-stage)
// ─────────────────────────────────────────

function renderScoreDisplay(scoreObj, maxScore) {
    const { stage1, stage2, final } = scoreObj;
    const el = document.getElementById('score-display');
    const color = scoreColor(final, maxScore);
    const pct = final / maxScore;

    el.querySelector('#score-value').textContent = final.toFixed(2);
    el.querySelector('#score-value').style.color = color;
    el.querySelector('#score-max').textContent = `/ ${maxScore.toFixed(1)}`;

    const fill = el.querySelector('#score-bar-fill');
    fill.style.background = `linear-gradient(90deg, ${color}, ${color}99)`;
    fill.style.transform = 'scaleX(0)';
    fill.classList.remove('animate');
    requestAnimationFrame(() => requestAnimationFrame(() => {
        fill.style.transform = `scaleX(${pct})`;
        fill.classList.add('animate');
    }));

    // ── Stage breakdown panel ──
    const breakdown = document.getElementById('stage-breakdown');

    breakdown.innerHTML = `
      <div class="stage-grid">
        <div class="stage-item">
          <div class="stage-item__icon">⚖️</div>
          <div class="stage-item__label">Rule-Based</div>
          <div class="stage-item__score" style="color:#fbbf24">+${stage1.toFixed(2)}</div>
          <div class="stage-item__sub">Direct word match floor</div>
        </div>
        <div class="stage-vs">+</div>
        <div class="stage-item">
          <div class="stage-item__icon">📄</div>
          <div class="stage-item__label">Paper NLP Grade</div>
          <div class="stage-item__score" style="color:#38bdf8">+${stage2.toFixed(2)}</div>
          <div class="stage-item__sub">Jaccard·0.15 + Edit·0.05 + Cosine·0.15 + NormWC·0.15 + Semantic·0.50</div>
        </div>
      </div>
      <div class="stage-final-note">
        Final score = <strong>min(${maxScore.toFixed(0)}, ${stage1.toFixed(2)} + ${stage2.toFixed(2)}) = ${final.toFixed(2)}</strong>
        — scores are added and capped at the maximum possible mark.
      </div>
    `;
}

// ─────────────────────────────────────────
// RENDER — Plain English Explanation
// ─────────────────────────────────────────

function renderExplanation(explanation) {
    const body = document.getElementById('explanation-body');
    body.innerHTML = '';

    explanation.sections.forEach(s => {
        const div = document.createElement('div');
        div.className = 'explanation-section' + (s.sub ? ' explanation-section--sub' : '');
        div.innerHTML = `
            <span class="explanation-section__icon">${s.icon}</span>
            <span class="explanation-section__text">${s.text}</span>
        `;
        body.appendChild(div);
    });

    if (explanation.tips.length > 0) {
        const box = document.createElement('div');
        box.className = 'tips-box';
        box.innerHTML = `
            <div class="tips-box__title">💡 How to Improve</div>
            <ul>${explanation.tips.map(t => `<li>${t}</li>`).join('')}</ul>
        `;
        body.appendChild(box);
    }
}

// ─────────────────────────────────────────
// RENDER — SHAP Chart
// ─────────────────────────────────────────

function renderShap(shap, maxScore) {
    const body = document.getElementById('shap-body');
    const maxAbs = Math.max(0.001, ...Object.values(shap).map(Math.abs));
    const halfW = 50;

    const labels = {
        feat_avg_semantic: 'Avg. Semantic Match',
        feat_max_semantic: 'Peak Semantic Match',
        feat_anchors_covered: 'Key Concepts Covered',
        feat_avg_jaccard: 'Vocabulary Overlap',
        feat_avg_edit: 'Phrasing Similarity',
    };

    const tooltips = {
        feat_avg_semantic: 'How closely the overall meaning of your answer matched the reference',
        feat_max_semantic: 'Your best-matching sentence or phrase vs the reference',
        feat_anchors_covered: 'Fraction of key concepts from the ideal answer you addressed',
        feat_avg_jaccard: 'How many of the same words you used vs the reference',
        feat_avg_edit: 'How similar your sentence structure is to the reference',
    };

    body.innerHTML = Object.entries(shap).map(([key, val]) => {
        const pct = (val / maxAbs) * halfW;
        const isPos = val >= 0;
        const barCls = isPos ? 'shap-row__fill--pos' : 'shap-row__fill--neg';
        const left = isPos ? 50 : 50 + pct;
        const width = Math.abs(pct);
        const valLabel = (val >= 0 ? '+' : '') + val.toFixed(3);
        const color = isPos ? '#4ade80' : '#f87171';

        return `
          <div class="shap-row" data-tooltip="${tooltips[key] || ''}">
            <span class="shap-row__label">${labels[key] || key}</span>
            <div class="shap-row__bar-wrap">
              <div class="shap-row__mid"></div>
              <div class="shap-row__fill ${barCls}" style="left:${left}%;width:${width}%"></div>
            </div>
            <span class="shap-row__value" style="color:${color}">${valLabel}</span>
          </div>
        `;
    }).join('');
}

// ─────────────────────────────────────────
// RENDER — Anchors
// ─────────────────────────────────────────

function renderAnchors(anchors) {
    const wrap = document.getElementById('anchors-wrap');
    if (!anchors || !anchors.length) {
        wrap.innerHTML = '<span class="anchor-tag">No anchors extracted</span>';
        return;
    }
    wrap.innerHTML = anchors.map(a => `<span class="anchor-tag">${a}</span>`).join('');
}

// ─────────────────────────────────────────
// DEMO FORM — Single answer grading
// ─────────────────────────────────────────

const demoForm = document.getElementById('demo-form');
const resultPanel = document.getElementById('result-panel');
const spinner = document.getElementById('spinner');

demoForm.addEventListener('submit', e => {
    e.preventDefault();

    const ref = document.getElementById('ref-answer').value.trim();
    const stu = document.getElementById('stu-answer').value.trim();
    const maxSc = parseFloat(document.getElementById('max-score').value) || 5;

    if (!ref || !stu) {
        showToast('Please fill in both answer fields.', 'error');
        return;
    }

    spinner.classList.add('active');
    resultPanel.classList.remove('visible');

    setTimeout(() => {
        try {
            const result = gradeAnswer(ref, stu, maxSc);
            const { scoreObj, features, shap, explanation } = result;

            renderScoreDisplay(scoreObj, maxSc);
            renderExplanation(explanation);
            renderShap(shap, maxSc);
            renderAnchors(features.anchors);

            spinner.classList.remove('active');
            resultPanel.classList.add('visible');
            resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } catch (err) {
            spinner.classList.remove('active');
            showToast('Scoring error: ' + err.message, 'error');
            console.error(err);
        }
    }, 80);
});

// ─────────────────────────────────────────
// BATCH CSV SCORING
// ─────────────────────────────────────────

const uploadZone = document.getElementById('upload-zone');
const batchInput = document.getElementById('batch-file-input');
const batchResults = document.getElementById('batch-results');
const batchTableWrap = document.getElementById('batch-table-wrap');
const batchProgress = document.getElementById('batch-progress');

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) processCsvFile(file);
});

uploadZone.addEventListener('click', () => batchInput.click());
batchInput.addEventListener('change', () => {
    if (batchInput.files[0]) processCsvFile(batchInput.files[0]);
});

function processCsvFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file.', 'error');
        return;
    }

    batchProgress.textContent = `⏳ Parsing ${file.name}…`;
    batchProgress.classList.remove('hidden');
    batchResults.classList.add('hidden');

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: results => {
            const rows = results.data;
            const headers = results.meta.fields || [];

            const colRef = headers.find(h => /ref|desired|model|ideal|answer.*ref/i.test(h)) || headers[1];
            const colStu = headers.find(h => /student|stu|response|answer.*stu/i.test(h) && h !== colRef) || headers[2];
            const colQ = headers.find(h => /question|q$/i.test(h)) || headers[0];
            const maxScoreDefault = 5;

            if (!colRef || !colStu) {
                showToast('Could not find reference/student answer columns. See format hint.', 'error');
                batchProgress.classList.add('hidden');
                return;
            }

            let scored = [];
            let processed = 0;

            function scoreChunk(start) {
                const end = Math.min(start + 10, rows.length);
                for (let i = start; i < end; i++) {
                    const row = rows[i];
                    const ref = (row[colRef] || '').trim();
                    const stu = (row[colStu] || '').trim();
                    if (!ref || !stu) continue;

                    const res = gradeAnswer(ref, stu, maxScoreDefault);
                    scored.push({
                        question: (row[colQ] || '').slice(0, 100),
                        student: stu.slice(0, 120),
                        score: res.scoreObj.final,
                        stage1: res.scoreObj.stage1,
                        stage2: res.scoreObj.stage2,
                        maxScore: maxScoreDefault,
                        coverage: res.features.feat_anchors_covered,
                        semantic: res.features.feat_avg_semantic,
                    });
                    processed++;
                }

                batchProgress.textContent = `⏳ Scoring… ${processed} / ${rows.length}`;

                if (end < rows.length) setTimeout(() => scoreChunk(end), 10);
                else renderBatchTable(scored);
            }

            scoreChunk(0);
        },
        error: err => {
            showToast('CSV parse error: ' + err.message, 'error');
            batchProgress.classList.add('hidden');
        }
    });
}

let batchData = [];
let sortCol = 'score';
let sortAsc = false;

function renderBatchTable(data) {
    batchData = data;
    batchProgress.classList.add('hidden');
    batchResults.classList.remove('hidden');
    renderTable(sortCol, sortAsc);
    showToast(`Scored ${data.length} answers successfully!`);
}

function renderTable(col, asc) {
    const sorted = [...batchData].sort((a, b) => {
        const va = a[col], vb = b[col];
        if (typeof va === 'number') return asc ? va - vb : vb - va;
        return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });

    const cols = [
        { key: 'question', label: 'Question' },
        { key: 'student', label: 'Student Answer' },
        { key: 'score', label: 'Final Score ↕' },
        { key: 'stage1', label: 'Rule-Based' },
        { key: 'stage2', label: 'Paper NLP Grade' },
        { key: 'coverage', label: 'Concepts %' },
    ];

    batchTableWrap.innerHTML = `
      <div class="batch-table-wrap">
        <table class="batch-table" id="batch-tbl">
          <thead><tr>
            ${cols.map(c => `
              <th class="${c.key === col ? 'sorted' : ''}" data-col="${c.key}">
                ${c.label}
                <span class="sort-icon">${c.key === col ? (asc ? '↑' : '↓') : '↕'}</span>
              </th>
            `).join('')}
          </tr></thead>
          <tbody>
            ${sorted.map(row => {
        const chipClass = scoreColorClass(row.score, row.maxScore);
        const barPct = Math.round((row.score / row.maxScore) * 100);
        const covPct = Math.round(row.coverage * 100);
        return `
                  <tr>
                    <td style="max-width:180px;font-size:.82rem">${row.question || '—'}</td>
                    <td style="max-width:220px">${row.student}</td>
                    <td>
                      <span class="score-chip ${chipClass}">${row.score.toFixed(2)} / ${row.maxScore}</span>
                      <span class="mini-bar"><span class="mini-bar__fill" style="width:${barPct}%"></span></span>
                    </td>
                    <td style="color:#fbbf24;font-family:monospace">${row.stage1.toFixed(2)}</td>
                    <td style="color:#38bdf8;font-family:monospace">${row.stage2.toFixed(2)}</td>
                    <td>${covPct}%</td>
                  </tr>
                `;
    }).join('')}
          </tbody>
        </table>
      </div>
      <div style="margin-top:12px;display:flex;gap:10px;justify-content:flex-end;">
        <button class="btn btn--ghost btn--sm" id="export-csv-btn">⬇ Export Results</button>
      </div>
    `;

    batchTableWrap.querySelectorAll('th[data-col]').forEach(th => {
        th.addEventListener('click', () => {
            const newCol = th.dataset.col;
            if (newCol === sortCol) sortAsc = !sortAsc;
            else { sortCol = newCol; sortAsc = false; }
            renderTable(sortCol, sortAsc);
        });
    });

    document.getElementById('export-csv-btn')?.addEventListener('click', exportCsv);
}

function exportCsv() {
    const headers = ['question', 'student_answer', 'final_score', 'stage1_rule_based', 'stage2_paper_nlp_grade', 'max_score', 'concepts_covered_%'];
    const rows = batchData.map(r => [
        `"${(r.question || '').replace(/"/g, '""')}"`,
        `"${r.student.replace(/"/g, '""')}"`,
        r.score.toFixed(3),
        r.stage1.toFixed(3),
        r.stage2.toFixed(3),
        r.maxScore,
        Math.round(r.coverage * 100),
    ].join(','));

    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'asag_results.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Results exported!');
}

// ─────────────────────────────────────────
// Sample Q&A prefill
// ─────────────────────────────────────────

document.getElementById('try-sample')?.addEventListener('click', () => {
    document.getElementById('ref-answer').value =
        'Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. Chlorophyll in the chloroplasts absorbs light energy which drives the conversion of CO2 and water into sugar and releases oxygen as a byproduct.';
    document.getElementById('stu-answer').value =
        'Plants make food using sunlight. They take in CO2 and water and produce oxygen. This happens in the leaves where chlorophyll is present.';
    document.getElementById('max-score').value = '5';
    document.getElementById('ref-answer').scrollIntoView({ behavior: 'smooth' });
});
