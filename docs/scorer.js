/**
 * scorer.js — In-browser ASAG Scoring Engine  (Two-Stage)
 *
 * Stage 1 — Rule-Based (floor)
 *   Scores by direct word matching: what fraction of the reference answer's
 *   important content words appear in the student answer?
 *   • Exact match  → 100% → full marks, always.
 *   • This score is the guaranteed minimum the student can never score below.
 *
 * Stage 2 — Paper's Grading Method  (PMC12171532)
 *   "Automated grading using NLP and semantic analysis" — Ahmad Ayaan & Kok-Why Ng
 *   Implements the exact published equations:
 *     Cnlp = min(max(0, wj·Sj + we·Se + wc·Sc + ww·Sw), 1)          ...(a)
 *     C    = min(max(0, wtf·Stf + (1-wtf)·Cnlp), 1)                  ...(b)
 *     F    = { 0  if Stf < 0.2                                        ...(c)
 *            { 1  if Stf >= 0.9 AND Sw >= 0.85
 *            { C  otherwise
 *     M    = ceil(min(F · T, T))                                       ...(d)
 *   Weights: Jaccard=0.15, EditSim=0.05, Cosine=0.15, NormWC=0.15, Semantic=0.50
 *
 * Final Score = min(maxScore, stage1 + stage2)
 *   The paper grade can only raise the bar — rule-based acts as a floor.
 */

// ─────────────────────────────────────────────────────────────
// 1. TEXT UTILITIES
// ─────────────────────────────────────────────────────────────

const STOPWORDS = new Set([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'no',
    'that', 'this', 'these', 'those', 'which', 'who', 'whom', 'what', 'how', 'when', 'where',
    'why', 'if', 'because', 'while', 'although', 'though', 'since', 'unless', 'until', 'than',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs', 'its', 'all', 'each',
    'any', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too',
    'very', 'just', 'about', 'above', 'after', 'again', 'also', 'back', 'being', 'between',
    'each', 'here', 'however', 'into', 'like', 'many', 'make', 'much', 'now', 'only', 'other',
    'our', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'm', 'o', 're', 've', 'y'
]);

function tokenize(text) {
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
}

function tokenizeFiltered(text) {
    return tokenize(text).filter(t => !STOPWORDS.has(t));
}

// ─────────────────────────────────────────────────────────────
// 2. JACCARD SIMILARITY
// ─────────────────────────────────────────────────────────────

function jaccardSimilarity(text1, text2) {
    const set1 = new Set(tokenize(text1));
    const set2 = new Set(tokenize(text2));
    if (set1.size === 0 || set2.size === 0) return 0;
    const intersection = [...set1].filter(t => set2.has(t)).length;
    const union = new Set([...set1, ...set2]).size;
    return intersection / union;
}

// ─────────────────────────────────────────────────────────────
// 3. EDIT DISTANCE (Normalised Levenshtein)
// ─────────────────────────────────────────────────────────────

function editSimilarity(text1, text2) {
    const a = text1.toLowerCase();
    const b = text2.toLowerCase();
    if (!a && !b) return 1;
    if (!a || !b) return 0;
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, (_, i) =>
        Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
    );
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            dp[i][j] = a[i - 1] === b[j - 1]
                ? dp[i - 1][j - 1]
                : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
        }
    }
    return 1 - dp[m][n] / Math.max(m, n);
}

// ─────────────────────────────────────────────────────────────
// 4. TF-IDF COSINE SIMILARITY
// Used only for anchor-keyphrase extraction scoring.
// NOTE: For the paper's Sc and Stf we use tfCosineSim (section 4B)
// because TF-IDF on a 2-doc corpus zeros out shared words (IDF→0),
// which gives ~0 similarity for two very similar texts.
// ─────────────────────────────────────────────────────────────

function buildTfVector(tokens) {
    const freq = {};
    for (const t of tokens) freq[t] = (freq[t] || 0) + 1;
    const total = tokens.length || 1;
    const tf = {};
    for (const t in freq) tf[t] = freq[t] / total;
    return tf;
}

function tfidfCosineSim(text1, text2) {
    const toks1 = tokenizeFiltered(text1);
    const toks2 = tokenizeFiltered(text2);
    if (!toks1.length || !toks2.length) return 0;

    const tf1 = buildTfVector(toks1);
    const tf2 = buildTfVector(toks2);

    // IDF: using both documents as the corpus
    const allTerms = new Set([...Object.keys(tf1), ...Object.keys(tf2)]);
    const idf = {};
    for (const t of allTerms) {
        const docCount = (tf1[t] ? 1 : 0) + (tf2[t] ? 1 : 0);
        idf[t] = Math.log(3 / (1 + docCount)); // smoothed idf over 2-doc corpus
    }

    // TF-IDF vectors
    let dot = 0, mag1 = 0, mag2 = 0;
    for (const t of allTerms) {
        const v1 = (tf1[t] || 0) * idf[t];
        const v2 = (tf2[t] || 0) * idf[t];
        dot += v1 * v2;
        mag1 += v1 * v1;
        mag2 += v2 * v2;
    }
    if (!mag1 || !mag2) return 0;
    return Math.max(0, Math.min(1, dot / (Math.sqrt(mag1) * Math.sqrt(mag2))));
}

// ─────────────────────────────────────────────────────────────
// 4B. PLAIN TF COSINE SIMILARITY  (no IDF weighting)
// Used for the paper's Sc (cosine) and Stf (semantic approx).
//
// Why: TF-IDF on a 2-doc corpus assigns IDF=0 to every word
// that appears in BOTH documents (log(3/3)=0), so two very
// similar answers end up with cosine≈0. Plain TF cosine rewards
// shared vocabulary and correctly gives high similarity when
// student and reference answers share most of their words.
// ─────────────────────────────────────────────────────────────

function tfCosineSim(text1, text2) {
    const toks1 = tokenize(text1);
    const toks2 = tokenize(text2);
    if (!toks1.length || !toks2.length) return 0;

    // Raw term frequency (count) vectors
    const tf1 = {}, tf2 = {};
    for (const t of toks1) tf1[t] = (tf1[t] || 0) + 1;
    for (const t of toks2) tf2[t] = (tf2[t] || 0) + 1;

    const allTerms = new Set([...Object.keys(tf1), ...Object.keys(tf2)]);
    let dot = 0, mag1 = 0, mag2 = 0;
    for (const t of allTerms) {
        const v1 = tf1[t] || 0;
        const v2 = tf2[t] || 0;
        dot += v1 * v2;
        mag1 += v1 * v1;
        mag2 += v2 * v2;
    }
    if (!mag1 || !mag2) return 0;
    return Math.max(0, Math.min(1, dot / (Math.sqrt(mag1) * Math.sqrt(mag2))));
}

// ─────────────────────────────────────────────────────────────
// 5. ANCHOR EXTRACTION (KeyBERT-like TF-IDF keyphrases)
// ─────────────────────────────────────────────────────────────

function ngramTokens(tokens, n) {
    const result = [];
    for (let i = 0; i <= tokens.length - n; i++) {
        result.push(tokens.slice(i, i + n).join(' '));
    }
    return result;
}

function extractAnchors(text, numAnchors = 5) {
    if (!text || !text.trim()) return [];
    const tokens = tokenizeFiltered(text);
    const candidates = [
        ...ngramTokens(tokens, 1),
        ...ngramTokens(tokens, 2),
        ...ngramTokens(tokens, 3),
    ];

    // Score by TF (within text) — simple frequency heuristic like top-n keyphrases
    const freq = {};
    for (const c of candidates) freq[c] = (freq[c] || 0) + 1;

    // Prefer longer keyphrases (like KeyBERT's use_maxsum)
    const scored = Object.entries(freq)
        .map(([phrase, count]) => ({ phrase, score: count * (1 + 0.3 * (phrase.split(' ').length - 1)) }))
        .sort((a, b) => b.score - a.score);

    // Deduplicate by removing phrases whose words are all contained in a higher-ranked phrase
    const selected = [];
    for (const { phrase } of scored) {
        const words = new Set(phrase.split(' '));
        const dominated = selected.some(s => {
            const sw = new Set(s.split(' '));
            return [...words].every(w => sw.has(w));
        });
        if (!dominated) selected.push(phrase);
        if (selected.length >= numAnchors) break;
    }
    return selected;
}

// ─────────────────────────────────────────────────────────────
// 6. COMPUTE FEATURES  (exactly mirrors semantic_mapping.py)
// ─────────────────────────────────────────────────────────────

function computeFeatures(referenceAnswer, studentAnswer, numAnchors = 5) {
    const anchors = extractAnchors(referenceAnswer, numAnchors);
    if (!anchors.length || !studentAnswer.trim()) {
        return {
            feat_avg_semantic: 0, feat_max_semantic: 0,
            feat_anchors_covered: 0, feat_avg_jaccard: 0, feat_avg_edit: 0,
            anchors,
        };
    }

    // Per-anchor scores — use plain TF cosine (tfCosineSim) so shared
    // words between the student answer and anchor are correctly rewarded.
    const COVERAGE_THRESHOLD = 0.35;

    const semanticScores = anchors.map(a => tfCosineSim(studentAnswer, a));
    const jaccardScores = anchors.map(a => jaccardSimilarity(studentAnswer, a));
    const editScores = anchors.map(a => editSimilarity(studentAnswer, a));

    const avg = arr => arr.reduce((s, v) => s + v, 0) / (arr.length || 1);
    const max = arr => Math.max(...arr);

    return {
        feat_avg_semantic: avg(semanticScores),
        feat_max_semantic: max(semanticScores),
        feat_anchors_covered: semanticScores.filter(s => s >= COVERAGE_THRESHOLD).length / anchors.length,
        feat_avg_jaccard: avg(jaccardScores),
        feat_avg_edit: avg(editScores),
        anchors,
        semanticScores, jaccardScores, editScores,
    };
}

// ─────────────────────────────────────────────────────────────
// 7A. STAGE 1 — RULE-BASED DIRECT WORD MATCH (Floor / Minimum)
//
// Logic:
//   1. Exact string match  → 100% → full marks immediately.
//   2. Otherwise: what % of the reference's content words (no stopwords)
//      are present anywhere in the student answer?
//   This is a recall-oriented score: did the student mention everything
//   the teacher wrote?
// ─────────────────────────────────────────────────────────────

function ruleBasedScore(referenceAnswer, studentAnswer, maxScore = 5) {
    // Guard: empty answers get zero
    if (!referenceAnswer.trim() || !studentAnswer.trim()) return 0;

    // ── Exact match → full marks ─────────────────────────────
    const refNorm = referenceAnswer.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    const stuNorm = studentAnswer.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    if (refNorm === stuNorm) return maxScore;

    // ── Word-level recall ────────────────────────────────────
    // Content words in reference (the target the student must hit)
    const refWords = tokenizeFiltered(referenceAnswer);
    if (refWords.length === 0) return 0;

    // All words (with stopwords) in the student's answer – broad match
    const stuAllWords = new Set(tokenize(studentAnswer));

    // Count how many reference content words appear in student answer
    const matchedCount = refWords.filter(w => stuAllWords.has(w)).length;
    const wordRecall = matchedCount / refWords.length;          // 0–1

    // ── Phrase-level bonus ───────────────────────────────────
    // Also check for whole unique bigrams / phrases from reference in student
    // (e.g. "carbon dioxide" should count as a unit).
    const refBigrams = ngramTokens(tokenizeFiltered(referenceAnswer), 2);
    const stuText = stuNorm;
    const bigramHits = refBigrams.filter(bg => stuText.includes(bg)).length;
    const bigramBonus = refBigrams.length > 0
        ? (bigramHits / refBigrams.length) * 0.15  // up to +15% boost
        : 0;

    const ruleRatio = Math.min(1, wordRecall + bigramBonus);
    return ruleRatio * maxScore;
}

// ─────────────────────────────────────────────────────────────
// 7B. STAGE 2 — PAPER'S GRADING METHOD  (PMC12171532)
// "Automated grading using NLP and semantic analysis"
// Ahmad Ayaan & Kok-Why Ng
// ─────────────────────────────────────────────────────────────

/**
 * normalizedWordCount(ref, stu)
 * Sw = reference_keywords / student_keywords, capped at 1.
 * Paper: "calculated by dividing the sample answer's keywords
 *         by the student's answer's keywords."
 */
function normalizedWordCount(referenceAnswer, studentAnswer) {
    const refKw = tokenizeFiltered(referenceAnswer);
    const stuKw = tokenizeFiltered(studentAnswer);
    if (stuKw.length === 0) return 0;
    return Math.min(1.0, refKw.length / stuKw.length);
}

/**
 * paperGradingScore(ref, stu, maxScore)
 *
 * Implements the EXACT equations from the paper:
 *   Weights: wj=0.15, we=0.05, wc=0.15, ww=0.15, wtf=0.50
 *
 *   Cnlp = min(max(0, wj·Sj + we·Se + wc·Sc + ww·Sw), 1)     ...(a)
 *   C    = min(max(0, wtf·Stf + (1-wtf)·Cnlp), 1)             ...(b)
 *   F    = { 0  if Stf < 0.2                                   ...(c)
 *          { 1  if Stf >= 0.9 AND Sw >= 0.85
 *          { C  otherwise
 *   M    = ceil(min(F · T, T))                                  ...(d)
 *
 * Returns F * maxScore (float) for additive combination with stage 1.
 * Note: Stf uses TF-IDF cosine as a surrogate for Universal Sentence Encoder.
 */
function paperGradingScore(referenceAnswer, studentAnswer, maxScore = 5) {
    if (!referenceAnswer.trim() || !studentAnswer.trim()) return 0;

    const Sj = jaccardSimilarity(referenceAnswer, studentAnswer);    // wj = 0.15
    const Se = editSimilarity(referenceAnswer, studentAnswer);        // we = 0.05
    const Sc = tfCosineSim(referenceAnswer, studentAnswer);           // wc = 0.15  (word-freq cosine)
    const Sw = normalizedWordCount(referenceAnswer, studentAnswer);   // ww = 0.15
    const Stf = tfCosineSim(referenceAnswer, studentAnswer);           // wtf= 0.50  (USE proxy via TF cosine)

    // Equation (a): Combined NLP base score
    const Cnlp = Math.min(1.0, Math.max(0.0, 0.15 * Sj + 0.05 * Se + 0.15 * Sc + 0.15 * Sw));

    // Equation (b): Confidence score
    const C = Math.min(1.0, Math.max(0.0, 0.50 * Stf + 0.50 * Cnlp));

    // Equation (c): Threshold rules
    let F;
    if (Stf < 0.2) F = 0.0;
    else if (Stf >= 0.9 && Sw >= 0.85) F = 1.0;
    else F = C;

    // Return F * maxScore (equation d uses ceil for integer marks;
    // we return float so the additive logic can apply it smoothly)
    return F * maxScore;
}

// ─────────────────────────────────────────────────────────────
// 7C. FINAL SCORE = min(maxScore, stage1 + stage2)
// Rule-based sets the floor; paper method adds semantic credit.
// ─────────────────────────────────────────────────────────────

function predictScore(referenceAnswer, studentAnswer, features, maxScore = 5) {
    const stage1 = ruleBasedScore(referenceAnswer, studentAnswer, maxScore);
    const stage2 = paperGradingScore(referenceAnswer, studentAnswer, maxScore);
    const final = Math.min(maxScore, stage1 + stage2);
    return { stage1, stage2, final };
}

// ─────────────────────────────────────────────────────────────
// 8. SHAP-STYLE ATTRIBUTIONS
// Attribution = weight * (feature_value - class_baseline)
// Baseline set to 0.40 (average student on Mohler dataset)
// ─────────────────────────────────────────────────────────────

// WEIGHTS used by shapValues for attribution display.
// Mapped to the paper's published weights (PMC12171532):
//   Semantic (USE)       wtf = 0.50  → feat_avg_semantic
//   Cosine               wc  = 0.15  → feat_max_semantic
//   Normalized Word Count ww = 0.15  → feat_anchors_covered
//   Jaccard              wj  = 0.15  → feat_avg_jaccard
//   Edit Similarity      we  = 0.05  → feat_avg_edit
const WEIGHTS = {
    feat_avg_semantic: 0.50,
    feat_max_semantic: 0.15,
    feat_anchors_covered: 0.15,
    feat_avg_jaccard: 0.15,
    feat_avg_edit: 0.05,
};

const BASELINE = 0.40;

function shapValues(features, maxScore = 5) {
    const scale = maxScore;
    return {
        feat_avg_semantic: (features.feat_avg_semantic - BASELINE) * WEIGHTS.feat_avg_semantic * scale,
        feat_max_semantic: (features.feat_max_semantic - BASELINE) * WEIGHTS.feat_max_semantic * scale,
        feat_anchors_covered: (features.feat_anchors_covered - BASELINE) * WEIGHTS.feat_anchors_covered * scale,
        feat_avg_jaccard: (features.feat_avg_jaccard - BASELINE) * WEIGHTS.feat_avg_jaccard * scale,
        feat_avg_edit: (features.feat_avg_edit - BASELINE) * WEIGHTS.feat_avg_edit * scale,
    };
}

// ─────────────────────────────────────────────────────────────
// 9. PLAIN-ENGLISH EXPLANATION  (mirrors Python _generate_plain_english_explanation)
// ─────────────────────────────────────────────────────────────

function generateExplanation(scoreObj, features, shapVals, maxScore = 5) {
    const score = scoreObj.final;
    const { feat_avg_semantic: avgSem, feat_max_semantic: maxSem,
        feat_anchors_covered: coverage, feat_avg_jaccard: jaccard,
        feat_avg_edit: editSim } = features;

    const pct = score / maxScore;
    const coveredPct = Math.round(coverage * 100);
    const sections = [];

    // ── Overall verdict
    let overall;
    if (pct >= 0.85)
        overall = `Your answer is excellent! You scored <strong>${score.toFixed(2)} out of ${maxScore.toFixed(1)}</strong>, placing you among the top performers.`;
    else if (pct >= 0.65)
        overall = `Your answer is good. You scored <strong>${score.toFixed(2)} out of ${maxScore.toFixed(1)}</strong>. You covered most of what was expected, with a little room to improve.`;
    else if (pct >= 0.40)
        overall = `Your answer is partially correct. You scored <strong>${score.toFixed(2)} out of ${maxScore.toFixed(1)}</strong>. You got some important points but missed several key ideas.`;
    else
        overall = `Your answer needs significant improvement. You scored <strong>${score.toFixed(2)} out of ${maxScore.toFixed(1)}</strong>. The answer is missing most of the core concepts expected.`;
    sections.push({ icon: pct >= 0.65 ? '🎯' : pct >= 0.40 ? '📋' : '📉', text: overall });

    // ── Key concept coverage
    let covText;
    if (coveredPct >= 70)
        covText = `You addressed approximately <strong>${coveredPct}%</strong> of the key concepts required in the ideal answer — this was a strong factor in your favour.`;
    else if (coveredPct >= 40)
        covText = `You addressed only about <strong>${coveredPct}%</strong> of the key concepts required. Missing the remaining concepts reduced your score (this factor ${shapVals.feat_anchors_covered > 0 ? 'raised' : 'lowered'} your mark).`;
    else
        covText = `You addressed very few (<strong>${coveredPct}%</strong>) of the key concepts expected in a complete answer. This was the biggest reason your score is low.`;
    sections.push({ icon: coveredPct >= 70 ? '✅' : coveredPct >= 40 ? '⚠️' : '❌', text: covText });

    // ── Semantic / meaning similarity
    let semText;
    if (avgSem >= 0.55)
        semText = `The overall meaning of your answer closely matched the expected answer (semantic similarity: <strong>${Math.round(avgSem * 100)}%</strong>). This shows you understood the topic well.`;
    else if (avgSem >= 0.35)
        semText = `The meaning of your answer partially matched the expected answer (semantic similarity: <strong>${Math.round(avgSem * 100)}%</strong>). You understood some concepts but your explanation could be more precise.`;
    else
        semText = `The meaning of your answer was quite different from the expected answer (semantic similarity: <strong>${Math.round(avgSem * 100)}%</strong>). The grader could not identify the core idea in your response.`;
    sections.push({ icon: avgSem >= 0.55 ? '✅' : avgSem >= 0.35 ? '⚠️' : '❌', text: semText });

    // ── Best phrase match
    let phraseText;
    if (maxSem >= 0.65)
        phraseText = `Your best phrase or sentence was a strong match (peak similarity: <strong>${Math.round(maxSem * 100)}%</strong>).`;
    else if (maxSem >= 0.45)
        phraseText = `Your best phrase was a partial match (peak similarity: <strong>${Math.round(maxSem * 100)}%</strong>). Try to be more specific.`;
    else
        phraseText = `Even your closest phrase had a low match (peak similarity: <strong>${Math.round(maxSem * 100)}%</strong>) — try to use the correct terminology.`;
    sections.push({ icon: '💬', text: phraseText, sub: true });

    // ── Vocabulary / keyword overlap
    let vocabText;
    if (jaccard >= 0.30)
        vocabText = `You used many of the same key words as the model answer (word overlap: <strong>${Math.round(jaccard * 100)}%</strong>), which helped your score.`;
    else if (jaccard >= 0.15)
        vocabText = `You used some of the expected vocabulary (word overlap: <strong>${Math.round(jaccard * 100)}%</strong>), but using more subject-specific terms would improve your mark.`;
    else
        vocabText = `Very few of the key words from the model answer appeared in your response (word overlap: <strong>${Math.round(jaccard * 100)}%</strong>). Make sure to use the correct terminology.`;
    sections.push({ icon: jaccard >= 0.30 ? '✅' : jaccard >= 0.15 ? '⚠️' : '❌', text: vocabText });

    // ── Phrasing / edit distance
    let phraseSim;
    if (editSim >= 0.55)
        phraseSim = `The way you phrased your answer was very similar to the expected answer (phrasing similarity: <strong>${Math.round(editSim * 100)}%</strong>).`;
    else if (editSim >= 0.30)
        phraseSim = `Your phrasing was somewhat similar to the model answer (phrasing similarity: <strong>${Math.round(editSim * 100)}%</strong>). Consider restructuring your sentences to be more concise and on-point.`;
    else
        phraseSim = `Your phrasing was quite different from the expected answer (phrasing similarity: <strong>${Math.round(editSim * 100)}%</strong>). This may indicate you expressed ideas in an unrelated way or went off-topic.`;
    sections.push({ icon: editSim >= 0.55 ? '✅' : editSim >= 0.30 ? '➡️' : '❌', text: phraseSim });

    // ── How to improve
    const tips = [];
    if (coveredPct < 70) tips.push('Re-read the question carefully and make sure you address ALL required points.');
    if (avgSem < 0.50) tips.push('Focus on expressing the core idea more clearly and directly.');
    if (jaccard < 0.25) tips.push('Use domain-specific vocabulary and keywords from your notes/textbook.');
    if (editSim < 0.40) tips.push('Write more structured, concise sentences that match the question\'s scope.');

    return { sections, tips };
}

// ─────────────────────────────────────────────────────────────
// 10. TOP-LEVEL API — called by app.js
// ─────────────────────────────────────────────────────────────

/**
 * gradeAnswer(ref, student, maxScore)
 *
 * Returns:
 *   scoreObj    = { stage1 (rule-based floor), stage2 (paper NLP+semantic), final }
 *   features    = raw NLP feature values
 *   shap        = per-feature SHAP-style attributions
 *   explanation = plain-English object { sections[], tips[] }
 */
function gradeAnswer(referenceAnswer, studentAnswer, maxScore = 5) {
    const features = computeFeatures(referenceAnswer, studentAnswer);
    const scoreObj = predictScore(referenceAnswer, studentAnswer, features, maxScore);
    const shap = shapValues(features, maxScore);
    const explanation = generateExplanation(scoreObj, features, shap, maxScore);
    return { scoreObj, maxScore, features, shap, explanation };
}

// Export for use as module OR global (GitHub Pages = global)
if (typeof module !== 'undefined') {
    module.exports = { gradeAnswer, ruleBasedScore, paperGradingScore, normalizedWordCount, computeFeatures, shapValues, generateExplanation };
}
