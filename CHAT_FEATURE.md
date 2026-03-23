# Chat Assistant Feature - Implementation Guide

## Overview

A natural language chat interface has been added to the ExplainGrade website that allows users to ask questions about student scores, statistics, and system explanations.

## What Was Added

### 1. **New Files**

- **`docs/chat.js`** — Complete chat assistant implementation
  - `ChatAssistant` class: Loads CSV data, processes queries, generates responses
  - `ChatUI` class: Manages the chat interface and user interactions

### 2. **Updated Files**

- **`docs/index.html`**
  - Added "Chat Assistant" section before footer
  - Added chat HTML structure with messages container and input area
  - Added navbar link to Chat section
  - Added script reference to `chat.js`

- **`docs/style.css`**
  - Added comprehensive chat styling
  - Responsive design for mobile/tablet
  - Dark/light theme support
  - Student card and result formatting styles

## Features

### Query Types Supported

#### 📊 **Statistics Queries**

- "How many students?" → Total student count
- "Who has the highest score?" → Highest scorer with details
- "Who has the lowest score?" → Lowest scorer
- "What's the average score?" → Mean score calculation
- "What's the median?" → Median score

#### 📈 **Threshold Queries**

- "How many students scored above 0.4?" → Find students above threshold
- "Students with scores less than 0.3?" → Find students below threshold
- Supports flexible phrasing (more than, below, greater than, etc.)

#### 👤 **Individual Student Queries**

- "Score for [name]?" → Find student by name
- "What about [email]?" → Find student by email
- "Performance of [student name]?" → Detailed student information
- Fuzzy name matching (partial name matches)

#### 🔧 **System Explanation Queries**

- "How does grading work?" → Explains two-stage pipeline
- "Explain the system?" → Full system overview
- "What is semantic similarity?" → Metric explanations
- "What's happening behind the scenes?" → System logic explanation

#### ❓ **Help Queries**

- "Help" → Shows available query types and examples
- "What can you do?" → Lists capabilities

### Response Formatting

**Student Search Results:**

```
Student Name (email@example.com)
Final Score: 0.403 / 5 (80.6%)

Metrics:
• Stage 1 (Rule-Based): 0.403
• Stage 2 (NLP Semantic): 0.0
• Semantic Similarity: 0.013
• Jaccard (Word Overlap): 0.09
```

**Statistics Results:**

- Color-coded score display
- Ranked student lists
- Detailed breakdowns

## Technical Details

### CSV File Loading

- Loads `grading_results.csv` from parent directory (`../grading_results.csv`)
- Expected columns: `email`, `name`, `final`, `stage1`, `stage2`, `semantic`, `jaccard`, `note`
- Data cached in memory after first load

### Query Processing

- Multiple regex patterns for flexible query matching
- Intent detection through keyword matching
- Fuzzy matching for student name/email searches
- Case-insensitive processing

### CSV Parsing

- Handles quoted fields and commas in values
- Flexible header matching
- Robust error handling

## Usage Examples

**Query Examples Users Can Try:**

1. "How many students?"
2. "Who has the highest score?"
3. "How many students scored above 0.3?"
4. "What's the score for Manikesh Kumar?"
5. "How does the grading system work?"
6. "Students less than 0.25?"
7. "Average score?"
8. "Help"

## Suggested Queries (UI Chips)

The chat interface includes helpful suggestion chips:

- "Who has the highest score?"
- "How many students scored below 0.3?"
- "What's the average score?"
- "Score for John Doe?"
- "How does the system work?"
- "Show all students above 0.4"

## Styling

### Color Scheme

- **High Scores** (≥70%): Green (#4ade80)
- **Mid Scores** (40-70%): Amber/Yellow (#fbbf24)
- **Low Scores** (<40%): Red (#f87171)
- Respects dark/light theme toggle

### Responsive Design

- Chat box scales to container width
- Message display optimized for mobile
- Suggestion chips arrange in grid layout
- Proper touch targets for mobile devices

## Known Behaviors

### CSV Loading

- First page load fetches CSV from `../grading_results.csv`
- Data remains in memory during session
- If CSV fails to load, chat shows: "Loading student data... Please try again in a moment."

### Query Matching

- Regex-based pattern matching (case-insensitive)
- Fuzzy name matching allows partial matches
- Email matching is substring-based
- Always shows top 20 results for large lists

### Score Interpretation

- Auto-generated feedback based on percentage
- Excellent: ≥70%
- Good attempt: 50-70%
- Partial understanding: 30-50%
- Needs improvement: <30%

## Browser Compatibility

- Works in all modern browsers (Chrome, Firefox, Safari, Edge)
- Requires JavaScript enabled
- No external dependencies for chat functionality
- Uses standard DOM APIs

## Installation/Deployment

1. Files are already in place:
   - `docs/chat.js` ✅
   - `docs/index.html` (updated) ✅
   - `docs/style.css` (updated) ✅

2. When deploying:
   - Ensure `grading_results.csv` is in the root directory
   - Or adjust the path in `chat.js` line ~54: `const response = await fetch('../grading_results.csv');`

3. For local testing:
   - Use a local web server (Python: `python -m http.server`)
   - Or serve with VS Code Live Server extension
   - Direct file opening won't work due to CORS restrictions

## Future Enhancements

Possible additions:

- Export chat history to text/PDF
- Advanced filtering (score ranges, alphabetical sorting)
- Visualization of score distribution
- Integration with grading history
- Persistent chat history (localStorage)
- Quick stats dashboard in chat
- Export student lists to CSV

## Troubleshooting

### Chat doesn't load data

- Check console (F12 → Console tab) for errors
- Verify `grading_results.csv` is accessible
- May need to serve files with a web server, not as local files

### Queries not recognized

- Try different phrasing
- Type "help" for full list of supported queries
- Check if student name/email is spelled correctly

### Styling issues

- Clear browser cache (Ctrl+Shift+Del)
- Try different theme (dark/light toggle)
- Check CSS file loaded correctly (F12 → Network tab)

## Code Structure

### ChatAssistant Class

- `constructor()` — Initialize and load data
- `loadData()` — Fetch CSV asynchronously
- `parseCSV()` — Parse CSV text to JSON objects
- `parseCSVLine()` — Handle quoted fields
- `getStatistics()` — Calculate aggregate stats
- `findStudentsAboveScore()` — Find students above threshold
- `findStudentsBelowScore()` — Find students below threshold
- `findStudent()` — Search by name or email with fuzzy matching
- `getSystemExplanation()` — Return system overview
- `processQuery()` — Main query processor
- `formatStudentDetails()` — Format individual student display
- `formatStudentList()` — Format list of students
- `getScoreInterpretation()` — Generate feedback based on score
- `getScoreClass()` — Determine CSS class for score color

### ChatUI Class

- `constructor()` — Initialize UI controller
- `init()` — Setup event listeners and welcome message
- `sendMessage()` — Process user input
- `addMessage()` — Add message to chat display

## Dependencies

None! The chat feature uses only vanilla JavaScript, no external libraries.
