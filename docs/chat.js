/**
 * chat.js — Chat Assistant for ExplainGrade
 * Handles queries about student scores, statistics, and system explanation
 */

"use strict";

class ChatAssistant {
  constructor() {
    this.students = [];
    this.maxScore = 5;
    this.isLoaded = false;
    this.loadData();
  }

  /**
   * Load CSV data from grading_results.csv
   * This assumes the CSV is available at ../grading_results.csv
   */
  async loadData() {
    try {
      const response = await fetch("./grading_results.csv");
      const csvText = await response.text();
      this.parseCSV(csvText);
      this.isLoaded = true;
      console.log(`Loaded ${this.students.length} student records`);
    } catch (error) {
      console.error("Failed to load student data:", error);
      this.isLoaded = false;
    }
  }

  /**
   * Parse CSV text and populate students array
   */
  parseCSV(csvText) {
    const lines = csvText.trim().split("\n");
    if (lines.length < 2) return;

    // Parse header
    const header = lines[0].split(",").map((h) => h.trim());
    const emailIdx = header.indexOf("email");
    const nameIdx = header.indexOf("name");
    const finalIdx = header.indexOf("final");
    const stage1Idx = header.indexOf("stage1");
    const stage2Idx = header.indexOf("stage2");
    const semanticIdx = header.indexOf("semantic");
    const jaccardIdx = header.indexOf("jaccard");

    // Parse data rows
    this.students = [];
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      const values = this.parseCSVLine(line);
      if (values.length <= 1) continue;

      const student = {
        email: values[emailIdx] || "",
        name: values[nameIdx] || "",
        final: parseFloat(values[finalIdx]) || 0,
        stage1: parseFloat(values[stage1Idx]) || 0,
        stage2: parseFloat(values[stage2Idx]) || 0,
        semantic: parseFloat(values[semanticIdx]) || 0,
        jaccard: parseFloat(values[jaccardIdx]) || 0,
      };

      if (student.email || student.name) {
        this.students.push(student);
      }
    }
  }

  /**
   * Parse CSV line handling quoted fields
   */
  parseCSVLine(line) {
    const result = [];
    let current = "";
    let insideQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      const nextChar = line[i + 1];

      if (char === '"') {
        if (insideQuotes && nextChar === '"') {
          // Escaped quote
          current += '"';
          i++;
        } else {
          // Toggle quote state
          insideQuotes = !insideQuotes;
        }
      } else if (char === "," && !insideQuotes) {
        // Field separator
        result.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  }

  /**
   * Calculate statistics from students data
   */
  getStatistics() {
    if (this.students.length === 0) return null;

    const scores = this.students.map((s) => s.final);
    const sorted = [...scores].sort((a, b) => a - b);

    return {
      count: this.students.length,
      highest: Math.max(...scores),
      lowest: Math.min(...scores),
      average: (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3),
      median: sorted[Math.floor(sorted.length / 2)].toFixed(3),
    };
  }

  /**
   * Find students with scores above threshold
   */
  findStudentsAboveScore(threshold) {
    const students = this.students.filter((s) => s.final > threshold);
    const sorted = students.sort((a, b) => b.final - a.final);
    return {
      count: sorted.length,
      students: sorted.slice(0, 20), // Limit to 20
    };
  }

  /**
   * Find students with scores below threshold
   */
  findStudentsBelowScore(threshold) {
    const students = this.students.filter((s) => s.final < threshold);
    const sorted = students.sort((a, b) => a.final - b.final);
    return {
      count: sorted.length,
      students: sorted.slice(0, 20), // Limit to 20
    };
  }

  /**
   * Find student by name or email (fuzzy matching)
   */
  findStudent(query) {
    const lowerQuery = query.toLowerCase();

    // Exact/partial email match
    let matches = this.students.filter((s) =>
      s.email.toLowerCase().includes(lowerQuery),
    );

    if (matches.length > 0) {
      return matches.length === 1 ? matches[0] : matches;
    }

    // Fuzzy name match
    matches = this.students.filter((s) => {
      const name = s.name.toLowerCase();
      // Simple fuzzy: check if all words in query appear in name
      const words = lowerQuery.split(/\s+/);
      return words.every((word) => name.includes(word));
    });

    if (matches.length > 0) {
      return matches.length === 1 ? matches[0] : matches;
    }

    return null;
  }

  /**
   * Generate system explanation (how the grading works)
   */
  getSystemExplanation() {
    return `
<strong>How ExplainGrade Works</strong>

ExplainGrade uses a <strong>two-stage grading pipeline</strong> to score student answers:

<strong>Stage 1 — Rule-Based Floor (Minimum Score)</strong>
This stage checks how many important words from the reference answer appear in the student's answer.
• If you nailed the exact answer → full marks
• This acts as a safety net — your score can never go below this stage

<strong>Stage 2 — NLP Semantic Scoring (Ceiling)</strong>
This stage measures how well your answer matches the reference at a deeper level using:
• <strong>Jaccard Similarity</strong> (0.15 weight): Word overlap
• <strong>Edit Similarity</strong> (0.05 weight): Character-level proximity  
• <strong>Cosine Similarity</strong> (0.15 weight): Semantic alignment
• <strong>Word Coverage</strong> (0.15 weight): How many key concepts you covered
• <strong>Semantic Similarity</strong> (0.50 weight): Overall meaning alignment using AI

<strong>Final Score = Stage1 + Stage2 (capped at max score)</strong>

The explanation you see after grading shows exactly which sentences helped or hurt your score, and why.
        `;
  }

  /**
   * Parse user query and generate response
   */
  processQuery(query) {
    if (!this.isLoaded || this.students.length === 0) {
      return "Loading student data... Please try again in a moment.";
    }

    const lowerQuery = query.toLowerCase();

    // ─── STATISTICS QUERIES ───

    if (
      /how\s+many\s+students/.test(lowerQuery) ||
      /total\s+students/.test(lowerQuery) ||
      /^how many/.test(lowerQuery)
    ) {
      const stats = this.getStatistics();
      return `There are <strong>${stats.count} students</strong> in total who have been graded.`;
    }

    if (
      /highest.*score|best.*student|top.*student|who.*score.*most/.test(
        lowerQuery,
      ) ||
      /max.*score/.test(lowerQuery)
    ) {
      const stats = this.getStatistics();
      const topStudent = this.students.reduce((prev, curr) =>
        curr.final > prev.final ? curr : prev,
      );
      return `
The <strong>highest score is ${topStudent.final.toFixed(3)}</strong> (out of 5).

<strong>Student:</strong> ${topStudent.name} (${topStudent.email})

Breakdown:
• <strong>Stage 1 (Rule-Based):</strong> ${topStudent.stage1.toFixed(3)}
• <strong>Stage 2 (NLP Semantic):</strong> ${topStudent.stage2.toFixed(3)}
• <strong>Semantic Score:</strong> ${topStudent.semantic.toFixed(3)}
• <strong>Jaccard (Word Overlap):</strong> ${topStudent.jaccard.toFixed(3)}
            `;
    }

    if (
      /lowest.*score|worst.*student|bottom.*student|minimum.*score/.test(
        lowerQuery,
      )
    ) {
      const stats = this.getStatistics();
      const bottomStudent = this.students.reduce((prev, curr) =>
        curr.final < prev.final ? curr : prev,
      );
      return `
The <strong>lowest score is ${bottomStudent.final.toFixed(3)}</strong> (out of 5).

<strong>Student:</strong> ${bottomStudent.name} (${bottomStudent.email})

The student may need additional support or clarification on the concepts.
            `;
    }

    if (/average|mean.*score/.test(lowerQuery)) {
      const stats = this.getStatistics();
      return `The <strong>average score is ${stats.average}</strong> (out of 5) across all ${stats.count} students.`;
    }

    if (/median/.test(lowerQuery)) {
      const stats = this.getStatistics();
      return `The <strong>median score is ${stats.median}</strong> (out of 5).`;
    }

    // ─── THRESHOLD QUERIES ───

    const aboveMatch = lowerQuery.match(
      /(?:more than|greater than|above|>\s*)(\d+(?:\.\d+)?)/i,
    );
    if (aboveMatch) {
      const threshold = parseFloat(aboveMatch[1]);
      const result = this.findStudentsAboveScore(threshold);
      if (result.count === 0) {
        return `No students scored above ${threshold}.`;
      }
      let response = `<strong>${result.count} students</strong> scored <strong>above ${threshold}</strong>:<br><br>`;
      response += this.formatStudentList(result.students.slice(0, 10));
      if (result.count > 10) {
        response += `<br>... and ${result.count - 10} more`;
      }
      return response;
    }

    const belowMatch = lowerQuery.match(
      /(?:less than|below|fewer than|<\s*)(\d+(?:\.\d+)?)/i,
    );
    if (belowMatch) {
      const threshold = parseFloat(belowMatch[1]);
      const result = this.findStudentsBelowScore(threshold);
      if (result.count === 0) {
        return `No students scored below ${threshold}.`;
      }
      let response = `<strong>${result.count} students</strong> scored <strong>below ${threshold}</strong>:<br><br>`;
      response += this.formatStudentList(result.students.slice(0, 10));
      if (result.count > 10) {
        response += `<br>... and ${result.count - 10} more`;
      }
      return response;
    }

    // ─── INDIVIDUAL STUDENT QUERIES ───

    if (
      /score.*for|marks.*of|grade.*of|performance.*of|result.*of/.test(
        lowerQuery,
      ) ||
      /(^|\s)(who is|what|score\s+)/i.test(lowerQuery)
    ) {
      // Extract potential student identifier
      const nameMatch = query.match(
        /(?:for|of|by)\s+([a-zA-Z\s]+?)(?:\s+\(|$|\?|,)/i,
      );
      const emailMatch = query.match(/(\S+@\S+)/);

      let searchQuery = nameMatch ? nameMatch[1].trim() : null;
      if (!searchQuery && emailMatch) searchQuery = emailMatch[1];

      if (!searchQuery) {
        // Try to extract any name-like words
        const words = query.split(/\s+/);
        searchQuery = words
          .filter((w) => /^[a-zA-Z]/.test(w) && w.length > 2)
          .join(" ")
          .replace(/[?,.]/g, "");
      }

      if (searchQuery) {
        const found = this.findStudent(searchQuery);

        if (!found) {
          return `I couldn't find a student named or emailed "${searchQuery}". Can you provide a full name or email address?`;
        }

        if (Array.isArray(found)) {
          let response = `I found ${found.length} students matching "${searchQuery}":<br><br>`;
          response += this.formatStudentList(found);
          return response;
        }

        return this.formatStudentDetails(found);
      }
    }

    // ─── SYSTEM & PROCESS QUERIES ───

    if (
      /how.*work|explain.*system|two.?stage|grading.*process|pipeline|stage 1|stage 2|rule.?based|semantic|nlp/.test(
        lowerQuery,
      ) ||
      /methodology|behind.*scenes|what.*happening/.test(lowerQuery)
    ) {
      return this.getSystemExplanation();
    }

    if (/help|what.*can.*do|what.*ask|instructions/.test(lowerQuery)) {
      return `
I can help you with:

📊 <strong>Statistics:</strong>
• "How many students?" 
• "What's the highest/lowest score?"
• "What's the average score?"
• "How many students scored above/below X?"

👤 <strong>Individual Student Info:</strong>
• "What's the score for [student name]?"
• "Show me the marks and explanation for [email]"
• "Performance of [student]?"

🔧 <strong>System Explanation:</strong>
• "How does the grading work?"
• "Explain the two-stage pipeline"
• "What is semantic similarity?"
• "What's happening behind the scenes?"

Just ask me anything about student performance or how the system works!
            `;
    }

    // ─── DEFAULT / UNKNOWN QUERIES ───

    return `
I didn't quite catch that. Here's what I can help with:

📊 <strong>Ask about scores:</strong> "Who has the highest score?" "How many students scored below 0.3?"

👤 <strong>Look up a student:</strong> "What's the score for John Doe?" or search by email

🔧 <strong>Understand the system:</strong> "How does grading work?" "Explain the pipeline"

Type "help" for more options, or ask me anything about your grading system!
        `;
  }

  /**
   * Format a single student's detailed information
   */
  formatStudentDetails(student) {
    const percentage = ((student.final / this.maxScore) * 100).toFixed(1);
    const scoreClass = this.getScoreClass(student.final);

    return `
<div class="student-card">
  <div class="student-header">
    <h4>${student.name}</h4>
    <span class="email">${student.email}</span>
  </div>
  
  <div class="score-breakdown">
    <div class="score-item">
      <strong>Final Score</strong>
      <span class="score-value ${scoreClass}">${student.final.toFixed(3)} / ${this.maxScore}</span>
      <span class="percentage">(${percentage}%)</span>
    </div>
    
    <div class="metrics">
      <div class="metric">
        <strong>Stage 1 (Rule-Based):</strong>
        <span>${student.stage1.toFixed(3)}</span>
      </div>
      <div class="metric">
        <strong>Stage 2 (NLP Semantic):</strong>
        <span>${student.stage2.toFixed(3)}</span>
      </div>
      <div class="metric">
        <strong>Semantic Similarity:</strong>
        <span>${student.semantic.toFixed(3)}</span>
      </div>
      <div class="metric">
        <strong>Jaccard (Word Overlap):</strong>
        <span>${student.jaccard.toFixed(3)}</span>
      </div>
    </div>
  </div>
  
  <div class="score-interpretation">
    ${this.getScoreInterpretation(student.final)}
  </div>
</div>
        `;
  }

  /**
   * Format a list of students
   */
  formatStudentList(students) {
    return students
      .map(
        (s, i) => `
<div class="student-row">
  <span class="rank">${i + 1}.</span>
  <span class="name">${s.name}</span>
  <span class="email">${s.email}</span>
  <span class="score-badge ${this.getScoreClass(s.final)}">${s.final.toFixed(3)}</span>
</div>
        `,
      )
      .join("");
  }

  /**
   * Determine score interpretation and feedback
   */
  getScoreInterpretation(score) {
    const percentage = (score / this.maxScore) * 100;
    if (percentage >= 70) {
      return "<strong>✅ Excellent performance!</strong> Your answer demonstrates strong understanding of the concepts.";
    } else if (percentage >= 50) {
      return "<strong>⚠️ Good attempt.</strong> Your answer covers key concepts but may need clarification in some areas.";
    } else if (percentage >= 30) {
      return "<strong>⚠️ Partial understanding.</strong> Your answer shows some understanding but misses important concepts.";
    } else {
      return "<strong>❌ Needs improvement.</strong> Review the concepts and try again.";
    }
  }

  /**
   * Get CSS class for score color coding
   */
  getScoreClass(score) {
    const percentage = (score / this.maxScore) * 100;
    if (percentage >= 70) return "score-high";
    if (percentage >= 40) return "score-mid";
    return "score-low";
  }
}

/**
 * Chat UI Controller
 */
class ChatUI {
  constructor() {
    this.assistant = new ChatAssistant();
    this.messagesContainer = null;
    this.inputField = null;
    this.sendButton = null;
    this.init();
  }

  init() {
    this.messagesContainer = document.getElementById("chat-messages");
    this.inputField = document.getElementById("chat-input");
    this.sendButton = document.getElementById("chat-send");

    if (!this.messagesContainer || !this.inputField || !this.sendButton) {
      console.error("Chat elements not found in DOM");
      return;
    }

    this.sendButton.addEventListener("click", () => this.sendMessage());
    this.inputField.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Add welcome message
    this.addMessage(
      "assistant",
      "Hi! 👋 I'm your ExplainGrade assistant. I can help you with student scores, statistics, and explain how the grading system works. What would you like to know?",
      true,
    );
  }

  /**
   * Send user message and get response
   */
  sendMessage() {
    const message = this.inputField.value.trim();
    if (!message) return;

    // Add user message
    this.addMessage("user", message);
    this.inputField.value = "";

    // Simulate thinking delay
    setTimeout(() => {
      const response = this.assistant.processQuery(message);
      this.addMessage("assistant", response, true);
    }, 300);
  }

  /**
   * Add message to chat UI
   */
  addMessage(role, text, isHTML = false) {
    const messageEl = document.createElement("div");
    messageEl.className = `chat-message chat-message--${role}`;

    const contentEl = document.createElement("div");
    contentEl.className = "chat-message__content";

    if (isHTML) {
      contentEl.innerHTML = text;
    } else {
      contentEl.textContent = text;
    }

    messageEl.appendChild(contentEl);
    this.messagesContainer.appendChild(messageEl);

    // Auto-scroll to bottom
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }
}

// Initialize chat when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    window.chatUI = new ChatUI();
  });
} else {
  window.chatUI = new ChatUI();
}
