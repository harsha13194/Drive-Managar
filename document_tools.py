# document_tools.py
# All document processing tools in one file

from typing import List, Dict
from ai_engine.nvidia_client import call_llm
from utils.file_processor import truncate_text, chunk_text
import json


# ============================================================================
# TOOL 1: Citation Extractor
# ============================================================================

CITATION_SYSTEM = """You are an academic citation extraction specialist.
Extract all references, citations, and bibliographic entries.

**📚 IN-TEXT CITATIONS**: Each unique in-text citation (Author, Year or [number] format)

**📖 BIBLIOGRAPHY / REFERENCES**: Numbered in standard format:
1. Author(s). (Year). Title. Journal/Publisher. DOI/URL if present.

**🔗 URLS & LINKS**: Any web links mentioned

**📊 CITATION STATISTICS**:
- Total citations found: X
- Most cited author: Name
- Publication year range: XXXX–XXXX"""


def run_citation_extractor(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        text = truncate_text(rec["text"], max_words=5000)
        prompt = f"Extract all citations from:\n\n**{rec['name']}**\n\n{text}"
        citations = call_llm(CITATION_SYSTEM, prompt, max_tokens=1500)
        results.append(f"## 📚 {rec['name']}\n\n{citations}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 2: Comparator
# ============================================================================

COMPARATOR_SYSTEM = """You are an expert document comparison analyst.
Compare the given documents and structure your response:
1. **Overview** — What each document covers
2. **Key Similarities** — Common themes, arguments, facts
3. **Key Differences** — Contrasting points, unique content
4. **Unique Insights** — What each document uniquely contributes
5. **Recommendation** — How to use these documents together
Cite document names in your analysis."""


def run_comparator(file_records: List[Dict], **kwargs) -> str:
    if len(file_records) < 2:
        return "⚠️ Please select at least 2 files to compare."

    doc_sections = []
    for rec in file_records:
        text = truncate_text(rec["text"], max_words=2500)
        doc_sections.append(f"=== {rec['name']} ({rec['word_count']:,} words) ===\n{text}")

    combined = "\n\n".join(doc_sections)
    prompt = f"Compare these {len(file_records)} documents:\n\n{combined}"
    return call_llm(COMPARATOR_SYSTEM, prompt, max_tokens=2000)


# ============================================================================
# TOOL 3: Entity Extractor
# ============================================================================

ENTITY_SYSTEM = """You are a Named Entity Recognition (NER) expert.
Extract and categorize all named entities:

**👤 People / Persons**: Names of individuals
**🏢 Organizations**: Companies, institutions, agencies
**📍 Locations**: Cities, countries, regions, addresses
**📅 Dates & Times**: Specific dates, periods, deadlines
**💰 Monetary Values**: Prices, budgets, financial figures
**📊 Quantities & Metrics**: Numbers, percentages, statistics
**📋 Miscellaneous**: Products, events, laws, documents

List each as bullets under its category."""


def run_entity_extractor(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        text = truncate_text(rec["text"], max_words=5000)
        prompt = f"Extract named entities from:\n\n**{rec['name']}**\n\n{text}"
        entities = call_llm(ENTITY_SYSTEM, prompt, max_tokens=1200)
        results.append(f"## 🏷️ {rec['name']}\n\n{entities}")

    return "\n\n---\n\n".join(results)



# ============================================================================
# TOOL 5: Important Points
# ============================================================================

IMPORTANT_POINTS_SYSTEM = """You are an expert at identifying and extracting key information.
Extract the most important points from the document in this format:

## 📌 Top Important Points

**Point 1: [Title]**
[2–3 sentence explanation]
Importance: ⭐⭐⭐⭐⭐

**Point 2: [Title]**
[2–3 sentence explanation]
Importance: ⭐⭐⭐⭐

... (continue for all important points)

## 🏆 Most Critical Point
[The single most important takeaway in 1–2 sentences]

## 📋 Quick Reference List
- Point 1 (one line)
- Point 2 (one line)
...

Rate each point's importance with 1–5 stars. Be comprehensive but concise."""


def run_important_points(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        chunks = chunk_text(rec["text"], chunk_words=4000)
        if len(chunks) == 1:
            result = call_llm(IMPORTANT_POINTS_SYSTEM, f"Document: **{rec['name']}**\n\n{chunks[0]}", max_tokens=1500)
        else:
            partial = []
            for i, chunk in enumerate(chunks, 1):
                p = call_llm(IMPORTANT_POINTS_SYSTEM, f"**{rec['name']}** Part {i}/{len(chunks)}\n\n{chunk}", max_tokens=700)
                partial.append(p)
            merge_sys = "Merge these partial important-points lists into one final list. Remove duplicates, keep all unique points, re-rank by importance."
            result = call_llm(merge_sys, "\n\n---\n\n".join(partial), max_tokens=1800)
        results.append(f"## 📌 {rec['name']}\n**({rec['word_count']:,} words)**\n\n{result}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 6: Insights
# ============================================================================

INSIGHTS_SYSTEM = """You are an expert document analyst who extracts deep insights.
Analyze the document and provide:

## 🔍 Core Insights
- 3–5 non-obvious insights that go beyond surface-level reading

## 💡 Hidden Patterns
- Patterns, trends, or recurring themes the reader might miss

## ⚠️ Critical Observations
- Gaps, contradictions, or areas needing attention

## 🎯 Strategic Takeaways
- Actionable conclusions drawn from the content

## 📊 Confidence Assessment
- How well-supported are the main claims? (High / Medium / Low with reason)

Be specific and analytical — avoid generic statements."""


def run_insights(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        chunks = chunk_text(rec["text"], chunk_words=4000)
        if len(chunks) == 1:
            prompt = f"Document: **{rec['name']}**\n\n{chunks[0]}"
            analysis = call_llm(INSIGHTS_SYSTEM, prompt, max_tokens=1500)
        else:
            partial = []
            for i, chunk in enumerate(chunks, 1):
                p = call_llm(INSIGHTS_SYSTEM, f"**{rec['name']}** Part {i}/{len(chunks)}\n\n{chunk}", max_tokens=800)
                partial.append(p)
            merge = (
                "Merge these partial analyses into one comprehensive insight report:\n\n"
                + "\n\n---\n\n".join(partial)
            )
            analysis = call_llm(
                "You are merging multiple analysis sections into one cohesive report. Remove redundancy. Keep all unique insights.",
                merge, max_tokens=1800
            )
        results.append(f"## 🔍 Insights: {rec['name']}\n**({rec['word_count']:,} words)**\n\n{analysis}")

    return "\n\n---\n\n".join(results)



# ============================================================================
# TOOL 8: QA Tool
# ============================================================================

QA_SYSTEM = """You are a precise document question-answering assistant.
Answer questions ONLY based on the provided documents.
If the answer is not in the documents, clearly state that.
Always cite which document your answer comes from.
Format: Answer → then → Source Document name."""


def run_qa_tool(file_records: List[Dict], question: str = "", **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."
    if not question.strip():
        return "⚠️ Please enter a question in the tool configuration."

    context_parts = []
    for rec in file_records:
        text = truncate_text(rec["text"], max_words=3000)
        context_parts.append(f"[Document: {rec['name']}]\n{text}")

    context = "\n\n".join(context_parts)
    prompt = f"Documents:\n\n{context}\n\nQuestion: {question}"
    return call_llm(QA_SYSTEM, prompt, max_tokens=1200)



# ============================================================================
# TOOL 10: Sentiment Analysis
# ============================================================================

SENTIMENT_SYSTEM = """You are an expert sentiment and tone analysis specialist.
Analyze the document and provide:
1. **Overall Sentiment**: Positive / Negative / Neutral / Mixed (with confidence %)
2. **Emotional Tone**: e.g. professional, urgent, optimistic, critical, neutral
3. **Key Sentiment Drivers**: Top 3–5 phrases driving the sentiment
4. **Section Breakdown**: Sentiment of intro, body, and conclusion
5. **Summary**: One sentence conclusion about the overall tone"""


def run_sentiment(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        text = truncate_text(rec["text"], max_words=5000)
        prompt = f"Analyze sentiment of:\n\n**{rec['name']}**\n\n{text}"
        analysis = call_llm(SENTIMENT_SYSTEM, prompt, max_tokens=1000)
        results.append(f"## 💬 {rec['name']}\n\n{analysis}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 11: Simplifier
# ============================================================================

SIMPLIFIER_SYSTEM = """You are a plain-language simplification expert.
Rewrite complex, technical, or legal text in simple, clear language (8th-grade reading level).

Rules:
- Replace jargon with plain words
- Break long sentences into short ones
- Use active voice
- Define unavoidable technical terms in parentheses
- Keep all original meaning intact

Format:
**SIMPLIFIED VERSION:**
[Your simplified text]

**KEY TERMS EXPLAINED:**
- term: plain-language definition

**WHAT THIS MEANS FOR YOU:**
[2–3 bullet point takeaways]"""

SIMPLIFIER_MERGE_SYSTEM = """You are merging multiple simplified sections into one clean document.
Combine the sections naturally, remove redundant term explanations, and produce one final
'WHAT THIS MEANS FOR YOU' section covering all parts."""


def run_simplifier(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        chunks = chunk_text(rec["text"], chunk_words=3000)

        if len(chunks) == 1:
            prompt = f"Simplify this document:\n\n**{rec['name']}**\n\n{chunks[0]}"
            simplified = call_llm(SIMPLIFIER_SYSTEM, prompt, max_tokens=1800)
        else:
            parts = []
            for i, chunk in enumerate(chunks, 1):
                prompt = f"**{rec['name']}** — Part {i}/{len(chunks)}\n\n{chunk}\n\nSimplify this section."
                part = call_llm(SIMPLIFIER_SYSTEM, prompt, max_tokens=900)
                parts.append(f"[Section {i}]\n{part}")

            merge_prompt = (
                f"Merge these simplified sections of **{rec['name']}** into one clean document:\n\n"
                + "\n\n".join(parts)
            )
            simplified = call_llm(SIMPLIFIER_MERGE_SYSTEM, merge_prompt, max_tokens=2000)

        results.append(f"## ✨ {rec['name']}\n**({rec['word_count']:,} words simplified)**\n\n{simplified}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 12: Summarizer
# ============================================================================

SUMMARIZER_SYSTEM = """You are an expert document summarizer.
Create clear, well-structured summaries capturing key ideas, main arguments, and important details.
Use bullet points for key findings. End with a one-sentence "Core Message"."""

SUMMARIZER_MERGE_SYSTEM = """You are an expert at synthesizing multiple summaries into one coherent summary.
Combine the provided partial summaries into a single, well-structured final summary.
Remove redundancy. Preserve all important points. End with a one-sentence "Core Message"."""


def run_summarizer(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        # Document Memory check:
        if rec.get("summary"):
            results.append(f"## 📄 {rec['name']} (Cached Summary)\n\n{rec['summary']}")
            continue

        chunks = chunk_text(rec["text"], chunk_words=4000)

        if len(chunks) == 1:
            prompt = f"Document: **{rec['name']}**\n\n{chunks[0]}\n\nProvide a comprehensive summary."
            summary = call_llm(SUMMARIZER_SYSTEM, prompt, max_tokens=1200)
        else:
            partial_summaries = []
            for i, chunk in enumerate(chunks, 1):
                prompt = (
                    f"Document: **{rec['name']}** — Part {i}/{len(chunks)}\n\n"
                    f"{chunk}\n\nSummarize this section."
                )
                partial = call_llm(SUMMARIZER_SYSTEM, prompt, max_tokens=600)
                partial_summaries.append(f"[Part {i}]\n{partial}")

            merge_prompt = (
                f"Document: **{rec['name']}** — {len(chunks)} sections\n\n"
                + "\n\n".join(partial_summaries)
                + "\n\nMerge these into one final comprehensive summary."
            )
            summary = call_llm(SUMMARIZER_MERGE_SYSTEM, merge_prompt, max_tokens=1500)

        # We will save this summary back to rec for memory storage in the caller
        rec["generated_summary"] = summary
        results.append(f"## 📄 {rec['name']}\n**({rec['word_count']:,} words extracted)**\n\n{summary}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 13: Timeline Extractor
# ============================================================================

TIMELINE_SYSTEM = """You are a timeline extraction specialist.
Extract all dated events and milestones. Format as:

**📅 TIMELINE**

| Date/Period | Event | Details |
|-------------|-------|---------|
| YYYY-MM-DD | Event name | Brief details |

After the table add:
**Key Milestones**: The 3–5 most significant events
**Time Span**: Earliest to latest date
**Gaps**: Notable time gaps in the timeline

If no explicit dates exist, extract implied sequences (first, then, finally)."""


def run_timeline_extractor(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        text = truncate_text(rec["text"], max_words=5000)
        prompt = f"Extract a chronological timeline from:\n\n**{rec['name']}**\n\n{text}"
        timeline = call_llm(TIMELINE_SYSTEM, prompt, max_tokens=1200)
        results.append(f"## 📅 {rec['name']}\n\n{timeline}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 14: Translator
# ============================================================================

TRANSLATOR_SYSTEM = """You are a professional document translator.
Translate accurately, preserving formatting, structure, technical terminology, tone and style.
Provide only the translated text — no commentary or preamble."""

LANGUAGES = [
    "Spanish", "French", "German", "Arabic", "Hindi", "Chinese (Simplified)",
    "Japanese", "Portuguese", "Russian", "Italian", "Korean", "Telugu",
    "Tamil", "Turkish", "Dutch", "Polish", "Swedish", "Bengali",
]


def run_translator(file_records: List[Dict], target_language: str = "Spanish", **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        chunks = chunk_text(rec["text"], chunk_words=3000)
        translated_parts = []

        for i, chunk in enumerate(chunks, 1):
            prompt = f"Translate to {target_language}:\n\n{chunk}"
            translated_parts.append(call_llm(TRANSLATOR_SYSTEM, prompt, max_tokens=3500, temperature=0.1))

        translated = "\n\n".join(translated_parts)
        results.append(f"## 🌐 {rec['name']} → {target_language}\n\n{translated}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# TOOL 15: Extract Text
# ============================================================================

def run_extract_text(file_records: List[Dict], **kwargs) -> str:
    if not file_records:
        return "⚠️ No files selected."

    results = []
    for rec in file_records:
        results.append(f"## 📄 Extracted Text: {rec['name']}\n\n{rec['text']}")

    return "\n\n---\n\n".join(results)


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

def run_tool(tool_name: str, file_records: List[Dict], **kwargs) -> str:
    """
    Main dispatcher function to run any tool by name.
    
    Args:
        tool_name: Name of the tool to run (e.g., 'summarizer', 'comparator')
        file_records: List of file records with text content
        **kwargs: Additional arguments specific to each tool
    
    Returns:
        String result from the tool
    """
    tools = {
        'citation_extractor': run_citation_extractor,
        'citation_extraction': run_citation_extractor,
        'comparator': run_comparator,
        'compare_documents': run_comparator,
        'entity_extractor': run_entity_extractor,
        'extract_entities': run_entity_extractor,
        'important_points': run_important_points,
        'insights': run_insights,
        'deep_insights': run_insights,
        'qa_tool': run_qa_tool,
        'qa_document': run_qa_tool,
        'sentiment': run_sentiment,
        'sentiment_analysis': run_sentiment,
        'simplifier': run_simplifier,
        'simplify_document': run_simplifier,
        'summarizer': run_summarizer,
        'summarize_document': run_summarizer,
        'timeline_extractor': run_timeline_extractor,
        'extract_timeline': run_timeline_extractor,
        'translator': run_translator,
        'translate_document': run_translator,
        'extract_text': run_extract_text,
    }
    
    if tool_name not in tools:
        # Fuzzy fallback mapping
        norm_name = tool_name.lower().replace(' ', '_').replace('-', '_')
        if norm_name in tools:
            return tools[norm_name](file_records, **kwargs)
        return f"⚠️ Unknown tool: {tool_name}. Available tools: {', '.join(tools.keys())}"
    
    return tools[tool_name](file_records, **kwargs)
