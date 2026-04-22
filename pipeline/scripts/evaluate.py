#!/usr/bin/env python3
"""
Cross-Scale TI Evaluation Suite
================================
45 questions spanning 12 behavioral categories mapped to 7 knowledge subcategories.
Runs identically against any Qwen3.5 model size for direct comparison.

Behavioral categories (from dataset description):
  1. Premise Challenge           2. Data Source Skepticism
  3. Multi-Signal Synthesis      4. Live Signal Interpretation
  5. Location Feasibility        6. Stakeholder Management
  7. Always-On Intelligence      8. Job Description Analysis
  9. Forward-Looking/Scenario    10. Commercial Translation
  11. Diagnostic Questioning     12. Epistemic Calibration

Knowledge subcategories (from lit review, mapped to Culshaw 2022):
  A. Labor Market Interpretation
  B. Compensation & Benefits Analysis
  C. Competitive Intelligence
  D. Data Quality & Methodological Judgment
  E. Strategic Workforce Planning
  F. Stakeholder Communication & Commercial Acumen
  G. Operational Intelligence Design

Usage:
    python evaluate.py --model merged_qwen3.5-9b --label "qwen3.5-9b-ft"
    python evaluate.py --adapter output/qwen3.5-4b/run_XXXX --base Qwen/Qwen3.5-4B --label "qwen3.5-4b-ft"
    python evaluate.py --model Qwen/Qwen3.5-9B --label "qwen3.5-9b-base"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# ── System Prompt (matches training data) ────────────────────
SYSTEM_PROMPT = (
    "You are an experienced talent market intelligence analyst specializing "
    "in workforce analytics and labor market interpretation. You provide "
    "nuanced, evidence-aware analysis that distinguishes signal from noise "
    "in talent market data. You reason through problems by evaluating data "
    "sources critically, considering context, and delivering actionable "
    "recommendations grounded in analytical rigor rather than vendor "
    "marketing or hype. You ask sharp, diagnostic questions before giving "
    "answers because the right question often matters more than the right "
    "data. When you don't know something, you say so. When a data source "
    "has limitations, you name them. Your goal is to make your stakeholders "
    "smarter about how they think about talent markets, not just to give "
    "them the answer they want to hear."
)

GENERAL_PROMPT = "You are a helpful assistant. Think carefully before responding."


# ── 45-Question Evaluation Suite ─────────────────────────────
# Each question is tagged with:
#   - category: behavioral category (1 of 12)
#   - knowledge_sub: knowledge subcategory (1 of 7, A-G)
#   - signals: expected reasoning indicators
#   - system: which system prompt to use

EVAL_SUITE = [
    # ── 1. PREMISE CHALLENGE (Knowledge: D, A) ────────────────
    {
        "id": "PC01", "category": "Premise Challenge", "knowledge_sub": "D",
        "name": "Vendor headcount claim",
        "what_it_tests": "Does the model push back on raw headcount = available talent?",
        "signals": ["total vs accessible", "filtering", "not all", "employer brand",
                     "competition", "actually available", "passive", "willing to move"],
        "prompt": "A vendor report says there are 50,000 data engineers in our metro area. "
                  "Our recruiting team says that proves there's plenty of talent and we just "
                  "need better sourcing. Does the data support their conclusion?",
    },
    {
        "id": "PC02", "category": "Premise Challenge", "knowledge_sub": "A",
        "name": "AI replacing all recruiters",
        "what_it_tests": "Does the model resist hype and provide nuanced pushback?",
        "signals": ["not quite", "nuanced", "careful", "depends", "however",
                     "augment", "judgment", "relationship"],
        "prompt": "AI is going to completely replace all talent acquisition teams within "
                  "3 years. We should stop hiring recruiters entirely and invest everything "
                  "in AI tools. Don't you agree?",
    },
    {
        "id": "PC03", "category": "Premise Challenge", "knowledge_sub": "E",
        "name": "Remote solves everything",
        "what_it_tests": "Does it challenge the assumption that remote eliminates location constraints?",
        "signals": ["time zone", "compliance", "tax", "culture", "management",
                     "not that simple", "depends on role", "collaboration"],
        "prompt": "We've decided to go fully remote so location doesn't matter anymore. "
                  "We can hire anyone anywhere for any role. Can you confirm this is the "
                  "right approach for our engineering expansion?",
    },

    # ── 2. DATA SOURCE SKEPTICISM (Knowledge: D) ─────────────
    {
        "id": "DS01", "category": "Data Source Skepticism", "knowledge_sub": "D",
        "name": "Comp data for board",
        "what_it_tests": "Does it question the P65 comp number methodology?",
        "signals": ["benchmark composition", "comparison set", "data freshness",
                     "base vs total", "methodology", "sample size", "who's in"],
        "prompt": "We pulled compensation data from our analytics platform and it shows our "
                  "software engineers are at the 65th percentile. Our CHRO wants to use this "
                  "in the board presentation next week as evidence that our compensation is "
                  "competitive. Should we go ahead?",
    },
    {
        "id": "DS02", "category": "Data Source Skepticism", "knowledge_sub": "D",
        "name": "LinkedIn data reliability",
        "what_it_tests": "Does it flag LinkedIn profile data limitations?",
        "signals": ["self-reported", "outdated", "bias", "not representative",
                     "passive", "sample", "completeness"],
        "prompt": "Our team scraped LinkedIn to build a talent map of ML engineers in the "
                  "Southeast. We found 8,200 profiles. How reliable is this as a basis for "
                  "our hiring strategy?",
    },
    {
        "id": "DS03", "category": "Data Source Skepticism", "knowledge_sub": "D",
        "name": "WEF report interpretation",
        "what_it_tests": "Does it interrogate aggregated headline claims?",
        "signals": ["who was surveyed", "methodology", "operationalized",
                     "industry mix", "self-reported", "context", "headline"],
        "prompt": "The World Economic Forum says 39% of core skills will be disrupted by 2030. "
                  "Our CEO wants to use this in our workforce strategy deck. What should I "
                  "tell her about this number?",
    },
    {
        "id": "DS04", "category": "Data Source Skepticism", "knowledge_sub": "D",
        "name": "BLS data for local decisions",
        "what_it_tests": "Does it flag limitations of federal data for local hiring?",
        "signals": ["lag", "classification", "local", "granularity",
                     "doesn't capture", "supplement", "cross-reference"],
        "prompt": "I've been using BLS Occupational Employment Statistics to benchmark our "
                  "local engineering salaries. Is this sufficient?",
    },

    # ── 3. MULTI-SIGNAL SYNTHESIS (Knowledge: C) ─────────────
    {
        "id": "MS01", "category": "Multi-Signal Synthesis", "knowledge_sub": "C",
        "name": "Competitor AI pivot",
        "what_it_tests": "Can it weave 3 signals into a coherent competitor narrative?",
        "signals": ["competitor pivot", "AI investment", "strategic bet",
                     "poaching risk", "connected signals", "implications"],
        "prompt": "Three things happened this month: our main competitor posted 15 new AI "
                  "engineering roles (up from 2/month), our senior ML engineer just got a "
                  "LinkedIn message from their recruiter, and the competitor's CEO gave a "
                  "keynote about 'AI-first transformation.' What's the picture here?",
    },
    {
        "id": "MS02", "category": "Multi-Signal Synthesis", "knowledge_sub": "C",
        "name": "Layoff + hiring signals",
        "what_it_tests": "Can it reconcile contradictory market signals?",
        "signals": ["restructuring", "not contradictory", "different skills",
                     "rebalancing", "growth area", "declining area", "net effect"],
        "prompt": "A major tech company just announced 2,000 layoffs in their cloud division "
                  "but simultaneously posted 500 new roles in AI/ML. Their stock went up 4%. "
                  "What does this tell us about the talent market?",
    },
    {
        "id": "MS03", "category": "Multi-Signal Synthesis", "knowledge_sub": "C",
        "name": "Three competitor signals",
        "what_it_tests": "Can it synthesize competitor intelligence from diverse data types?",
        "signals": ["pattern", "strategic", "timing", "implications",
                     "competitive response", "talent war"],
        "prompt": "Competitor A opened a new office in Austin. Competitor B just acquired a "
                  "small AI startup with 30 engineers. Competitor C raised their engineering "
                  "referral bonus from $5K to $15K. What's happening in our competitive landscape?",
    },

    {
        "id": "MS04", "category": "Multi-Signal Synthesis", "knowledge_sub": "C",
        "name": "M&A talent due diligence",
        "what_it_tests": "Can it identify talent risks in an acquisition context?",
        "signals": ["key person risk", "retention", "culture", "overlap",
                     "earn-out", "non-compete", "integration", "attrition"],
        "prompt": "We're acquiring a 200-person company. Their CTO and 3 VPs of Engineering "
                  "have 1-year non-competes but no retention bonuses. Their Glassdoor reviews "
                  "mention 'uncertainty about the acquisition.' What talent risks should we "
                  "flag for the deal team?",
    },

    # ── 4. LIVE SIGNAL INTERPRETATION (Knowledge: A, D) ──────
    {
        "id": "LS01", "category": "Live Signal Interpretation", "knowledge_sub": "A",
        "name": "LLM fine-tuning spike",
        "what_it_tests": "Does it evaluate base rates before reacting?",
        "signals": ["base rate", "small numbers", "200% from low base",
                     "structural vs transient", "who's posting", "monitor"],
        "prompt": "The weekly skills tracking feed shows that 'LLM fine-tuning' mentions in "
                  "our industry's job postings increased 200% quarter-over-quarter, from 15 "
                  "to 45 postings. Is this something we should act on or just monitor?",
    },
    {
        "id": "LS02", "category": "Live Signal Interpretation", "knowledge_sub": "A",
        "name": "Attrition spike interpretation",
        "what_it_tests": "Does it consider structural vs seasonal vs random variation?",
        "signals": ["baseline", "seasonal", "exit interview", "small sample",
                     "department", "manager", "pattern vs noise"],
        "prompt": "We just lost 4 senior engineers in the past 6 weeks. HR says attrition "
                  "is spiking. Should we panic?",
    },
    {
        "id": "LS03", "category": "Live Signal Interpretation", "knowledge_sub": "D",
        "name": "Glassdoor rating drop",
        "what_it_tests": "Does it question the signal quality of review platforms?",
        "signals": ["sample size", "recency", "selection bias", "disgruntled",
                     "trend vs snapshot", "compare to", "context"],
        "prompt": "Our Glassdoor rating dropped from 4.1 to 3.6 in two months. The VP of "
                  "People wants to launch an urgent employer brand campaign. Is this the "
                  "right response?",
    },

    # ── 5. LOCATION FEASIBILITY (Knowledge: A, E) ────────────
    {
        "id": "LF01", "category": "Location Feasibility", "knowledge_sub": "E",
        "name": "Denver vs SLC analytics team",
        "what_it_tests": "Does it structure a proper feasibility framework?",
        "signals": ["talent supply", "competition", "compensation",
                     "hiring velocity", "risk factors", "recommendation"],
        "prompt": "We're considering opening a 30-person analytics team in Denver versus "
                  "Salt Lake City. What should I be looking at to make this recommendation?",
    },
    {
        "id": "LF02", "category": "Location Feasibility", "knowledge_sub": "A",
        "name": "India GCC feasibility",
        "what_it_tests": "Does it go beyond cost arbitrage to address real risks?",
        "signals": ["attrition", "competition", "tier-1 vs tier-2", "ramp time",
                     "management overhead", "quality", "cultural"],
        "prompt": "Leadership wants to open a Global Capability Center in Bangalore for 100 "
                  "engineers. They see the salary savings and want to move fast. What should "
                  "I flag before they commit?",
    },
    {
        "id": "LF03", "category": "Location Feasibility", "knowledge_sub": "E",
        "name": "Nearshore vs offshore",
        "what_it_tests": "Does it structure a multi-factor comparison?",
        "signals": ["time zone", "cost", "talent depth", "attrition",
                     "IP protection", "communication", "trade-off"],
        "prompt": "We're debating between opening a development center in Mexico City or "
                  "expanding our existing team in Poland. How do I frame this comparison?",
    },

    # ── 6. STAKEHOLDER MANAGEMENT (Knowledge: F) ─────────────
    {
        "id": "SM01", "category": "Stakeholder Management", "knowledge_sub": "F",
        "name": "CEO Miami bias",
        "what_it_tests": "Does it coach on HOW to deliver the message, not just data?",
        "signals": ["don't say she's wrong", "frame the data", "alternative framing",
                     "political awareness", "present options", "anchor"],
        "prompt": "Our CEO is convinced we should build our new engineering center in Miami "
                  "because she went to a conference there and loved it. The talent data clearly "
                  "favors other cities for our engineering profiles. How do I handle this?",
    },
    {
        "id": "SM02", "category": "Stakeholder Management", "knowledge_sub": "F",
        "name": "Delivering bad comp news",
        "what_it_tests": "Does it frame the conversation strategically, not just factually?",
        "signals": ["frame", "options", "cost", "risk", "retention",
                     "competitive", "phased approach", "don't lead with"],
        "prompt": "Our compensation analysis shows we're 20% below market for data science "
                  "roles. The CFO has already said there's no budget for raises this year. "
                  "How do I present this?",
    },
    {
        "id": "SM03", "category": "Stakeholder Management", "knowledge_sub": "F",
        "name": "Hiring manager unrealistic timeline",
        "what_it_tests": "Does it help navigate the political dynamics?",
        "signals": ["expectation", "trade-off", "data", "alternatives",
                     "partner", "reframe", "what's realistic"],
        "prompt": "A hiring manager insists they need 10 senior ML engineers hired within "
                  "45 days. Market data says median time-to-fill for this role is 90+ days. "
                  "How do I handle this without damaging the relationship?",
    },

    # ── 7. ALWAYS-ON INTELLIGENCE (Knowledge: G) ─────────────
    {
        "id": "AO01", "category": "Always-On Intelligence", "knowledge_sub": "G",
        "name": "Build always-on capability",
        "what_it_tests": "Does it think in tiers, cadences, and delivery mechanisms?",
        "signals": ["tiered", "cadence", "weekly", "monthly", "quarterly",
                     "field alerts", "battlecards", "feedback loop", "capacity"],
        "prompt": "I've been doing ad-hoc talent intelligence projects for two years. My VP "
                  "wants me to propose an 'always-on' intelligence capability. What should "
                  "that look like?",
    },
    {
        "id": "AO02", "category": "Always-On Intelligence", "knowledge_sub": "G",
        "name": "Dashboard design",
        "what_it_tests": "Does it distinguish vanity metrics from actionable intelligence?",
        "signals": ["actionable", "so what", "decision", "audience",
                     "cadence", "threshold", "alert", "not just data"],
        "prompt": "HR wants me to build a 'talent intelligence dashboard.' They've listed "
                  "20 metrics they want to track. How should I approach this?",
    },
    {
        "id": "AO03", "category": "Always-On Intelligence", "knowledge_sub": "G",
        "name": "Competitive intel cadence",
        "what_it_tests": "Does it design a monitoring system, not just a one-time report?",
        "signals": ["monitoring", "triggers", "cadence", "sources",
                     "distribution", "action", "escalation"],
        "prompt": "The Chief People Officer wants ongoing competitive intelligence on our "
                  "top 5 talent competitors. How do I set this up sustainably as a team of one?",
    },

    # ── 8. JOB DESCRIPTION ANALYSIS (Knowledge: A, B) ────────
    {
        "id": "JD01", "category": "Job Description Analysis", "knowledge_sub": "A",
        "name": "Unicorn JD evaluation",
        "what_it_tests": "Does it show how each requirement shrinks the pool?",
        "signals": ["each requirement narrows", "pool size", "intersection",
                     "compensation mismatch", "must-have vs nice-to-have", "unicorn"],
        "prompt": "Our hiring manager wrote a job description for a 'Senior Full-Stack AI "
                  "Engineer' requiring React, Python, PyTorch, Kubernetes, 8+ years experience, "
                  "and a PhD. Compensation range is $160-190K. Can you help me evaluate this "
                  "against the market?",
    },
    {
        "id": "JD02", "category": "Job Description Analysis", "knowledge_sub": "B",
        "name": "Comp band too wide",
        "what_it_tests": "Does it diagnose the structural problem with wide bands?",
        "signals": ["too wide", "different roles", "equity", "internal compression",
                     "retention risk", "market rate", "split"],
        "prompt": "We have a single salary band of $120K-$200K for all 'Senior Engineer' roles "
                  "regardless of specialization. Is this a problem?",
    },

    {
        "id": "JD03", "category": "Job Description Analysis", "knowledge_sub": "B",
        "name": "Equity vs base trade-off",
        "what_it_tests": "Does it analyze the comp structure, not just the number?",
        "signals": ["equity", "vesting", "cash vs stock", "stage of company",
                     "risk profile", "total comp", "liquid vs illiquid"],
        "prompt": "A startup is offering our senior engineer candidate $140K base plus "
                  "$200K in stock options over 4 years. We're offering $185K base with no "
                  "equity. They say the startup offer is worth more. How do I evaluate this?",
    },

    # ── 9. FORWARD-LOOKING / SCENARIO (Knowledge: E) ─────────
    {
        "id": "FL01", "category": "Forward-Looking", "knowledge_sub": "E",
        "name": "Data engineering 2028",
        "what_it_tests": "Does it build scenarios rather than point predictions?",
        "signals": ["scenario", "can't predict", "structural forces",
                     "uncertain", "leading indicators", "monitor for"],
        "prompt": "What will the data engineering talent market look like in 2028? We need "
                  "to plan our hiring strategy.",
    },
    {
        "id": "FL02", "category": "Forward-Looking", "knowledge_sub": "E",
        "name": "AI impact on workforce size",
        "what_it_tests": "Does it resist giving a definitive number?",
        "signals": ["depends", "which roles", "augment vs replace", "timeline",
                     "scenario", "uncertainty", "adjacent roles"],
        "prompt": "How many fewer people will we need in 3 years because of AI? Give me a number.",
    },
    {
        "id": "FL03", "category": "Forward-Looking", "knowledge_sub": "E",
        "name": "Skills taxonomy future-proofing",
        "what_it_tests": "Does it address the limitations of static skill models?",
        "signals": ["half-life", "evolving", "adjacent skills", "capability",
                     "not static", "update cadence", "proxy"],
        "prompt": "We want to build a skills taxonomy that will be valid for the next 5 years. "
                  "What should we know before investing in this?",
    },

    # ── 10. COMMERCIAL TRANSLATION (Knowledge: F, B) ─────────
    {
        "id": "CT01", "category": "Commercial Translation", "knowledge_sub": "F",
        "name": "TTF increase for CFO",
        "what_it_tests": "Does it translate to dollars and business impact?",
        "signals": ["dollar", "cost of vacancy", "revenue delay",
                     "budget", "CFO language", "ROI", "options with costs"],
        "prompt": "Our time-to-fill for senior engineers has increased from 55 to 90 days "
                  "over the past year. How should I present this to our CFO?",
    },
    {
        "id": "CT02", "category": "Commercial Translation", "knowledge_sub": "B",
        "name": "Retention cost modeling",
        "what_it_tests": "Does it quantify the business case for retention investment?",
        "signals": ["replacement cost", "productivity ramp", "institutional knowledge",
                     "1.5x to 2x", "revenue impact", "team disruption"],
        "prompt": "Engineering turnover is at 18%. The CHRO wants to propose a $2M retention "
                  "package to the board. How do I build the business case?",
    },
    {
        "id": "CT03", "category": "Commercial Translation", "knowledge_sub": "F",
        "name": "TI ROI justification",
        "what_it_tests": "Does it frame TI value in business terms, not HR terms?",
        "signals": ["avoided cost", "speed", "quality of hire", "risk reduction",
                     "decision quality", "concrete example"],
        "prompt": "My VP asked me to justify the cost of our talent intelligence function. "
                  "What's the framework for demonstrating ROI?",
    },

    # ── 11. DIAGNOSTIC QUESTIONING (Knowledge: D, G) ─────────
    {
        "id": "DQ01", "category": "Diagnostic Questioning", "knowledge_sub": "D",
        "name": "Vague hiring request",
        "what_it_tests": "Does it ask scoping questions BEFORE providing analysis?",
        "signals": ["what role", "how many", "when", "where", "budget",
                     "what's driving", "before I can", "question", "?"],
        "prompt": "We need to hire more engineers. Can you help?",
    },
    {
        "id": "DQ02", "category": "Diagnostic Questioning", "knowledge_sub": "G",
        "name": "Ambiguous comp request",
        "what_it_tests": "Does it diagnose what they actually need before answering?",
        "signals": ["which roles", "what market", "base or total", "what decision",
                     "benchmark against", "question", "?"],
        "prompt": "Can you pull some comp data for us?",
    },
    {
        "id": "DQ03", "category": "Diagnostic Questioning", "knowledge_sub": "D",
        "name": "Talent mapping request",
        "what_it_tests": "Does it ask what the map is for before building it?",
        "signals": ["purpose", "what decision", "scope", "geography",
                     "seniority", "timeline", "question", "?"],
        "prompt": "We need a talent map for our product team.",
    },
    {
        "id": "DQ04", "category": "Diagnostic Questioning", "knowledge_sub": "G",
        "name": "Board deck data request",
        "what_it_tests": "Does it ask about audience and goal before pulling data?",
        "signals": ["audience", "what story", "what decision", "context",
                     "what have you shown before", "question", "?"],
        "prompt": "I need talent data for the board deck next week.",
    },

    # ── 12. EPISTEMIC CALIBRATION (Knowledge: D) ─────────────
    {
        "id": "EC01", "category": "Epistemic Calibration", "knowledge_sub": "D",
        "name": "Predict competitor hiring",
        "what_it_tests": "Does it admit it can't predict and suggest indicators instead?",
        "signals": ["can't predict", "don't know", "uncertain", "impossible",
                     "however", "leading indicators", "watch for"],
        "prompt": "Will our competitors hire more or fewer engineers next year?",
    },
    {
        "id": "EC02", "category": "Epistemic Calibration", "knowledge_sub": "D",
        "name": "Exact salary for role",
        "what_it_tests": "Does it provide a range with appropriate caveats?",
        "signals": ["range", "depends", "location", "experience", "total comp",
                     "approximate", "varies", "without more context"],
        "prompt": "What's the exact salary for a senior data scientist?",
    },
    {
        "id": "EC03", "category": "Epistemic Calibration", "knowledge_sub": "D",
        "name": "Whether to offshore",
        "what_it_tests": "Does it refuse to give a simple yes/no and explain why?",
        "signals": ["depends", "trade-off", "factors", "context needed",
                     "not a simple", "risk", "what are you optimizing"],
        "prompt": "Should we offshore our QA team? Yes or no?",
    },

    # ── CROSS-CUTTING: BREVITY TESTS ─────────────────────────
    {
        "id": "BV01", "category": "Brevity", "knowledge_sub": "A",
        "name": "Simple definition",
        "what_it_tests": "Does it give a concise answer to a simple question?",
        "signals": [],
        "max_words": 150,
        "prompt": "What does TI stand for in talent intelligence?",
    },
    {
        "id": "BV02", "category": "Brevity", "knowledge_sub": "A",
        "name": "Quick feasibility check",
        "what_it_tests": "Does it lead with a direct answer?",
        "signals": ["yes", "no", "depends", "likely", "unlikely"],
        "max_words": 250,
        "prompt": "Can we hire 5 data engineers in Austin in 3 months?",
    },

    # ── CROSS-CUTTING: SYSTEM PROMPT SWITCHING ───────────────
    {
        "id": "SW01", "category": "Switching", "knowledge_sub": "A",
        "name": "TI prompt activates domain",
        "what_it_tests": "With TI system prompt, does it respond as a TI analyst?",
        "signals": ["talent", "market", "data", "hiring", "compensation",
                     "competitor", "pipeline", "supply"],
        "system": SYSTEM_PROMPT,
        "prompt": "We need to grow our team by 20 people next year. Where should we start?",
    },
    {
        "id": "SW02", "category": "Switching", "knowledge_sub": "A",
        "name": "General prompt stays general",
        "what_it_tests": "With general prompt, does it NOT default to TI framing?",
        "signals": [],
        "anti_signals": ["talent intelligence", "labor market", "job postings",
                         "compensation benchmark", "talent pool"],
        "system": GENERAL_PROMPT,
        "prompt": "We need to grow our team by 20 people next year. Where should we start?",
    },

    # ── CROSS-CUTTING: GENERAL KNOWLEDGE PRESERVATION ────────
    {
        "id": "GK01", "category": "General Knowledge", "knowledge_sub": None,
        "name": "Medical knowledge preserved",
        "what_it_tests": "Did fine-tuning damage general knowledge?",
        "signals": ["virus", "bacteria", "antibiotic", "won't help"],
        "system": GENERAL_PROMPT,
        "prompt": "Should I take antibiotics for a cold?",
    },
    {
        "id": "GK02", "category": "General Knowledge", "knowledge_sub": None,
        "name": "Reasoning preserved",
        "what_it_tests": "Did fine-tuning damage basic reasoning?",
        "signals": ["5 cent", "$0.05", "0.05"],
        "system": GENERAL_PROMPT,
        "prompt": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. "
                  "How much does the ball cost?",
    },
]


# ── Model Loading ────────────────────────────────────────────
def load_model(model_path, base_model=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not base_model else base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model:
        print(f"  Loading base: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        print(f"  Loading adapter: {os.path.abspath(model_path)}")
        if not HAS_PEFT:
            print("ERROR: peft not installed")
            sys.exit(1)
        model = PeftModel.from_pretrained(model, os.path.abspath(model_path))
    else:
        print(f"  Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


# ── Generation ───────────────────────────────────────────────
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1].item() in self.stop_ids


def generate(model, tokenizer, prompt, system=SYSTEM_PROMPT, max_new_tokens=2048):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Build stop token list
    stop_strings = ["<|im_end|>", "<|im_start|>user", "<|im_start|>assistant"]
    stop_ids = set()
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.add(ids[-1])
    if tokenizer.eos_token_id:
        stop_ids.add(tokenizer.eos_token_id)

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens(list(stop_ids))]),
        )
    elapsed = time.time() - start
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    # Clean leaked turn markers
    for marker in ["\nuser\n", "\nUser:", "\nHuman:", "<|im_end|>", "<|im_start|>"]:
        if marker in response:
            response = response[:response.index(marker)]
    return response.strip(), new_tokens, elapsed


# ── Scoring ──────────────────────────────────────────────────
def signal_check(response, signals):
    r_lower = response.lower()
    found = [s for s in signals if s.lower() in r_lower]
    return found, len(found) / len(signals) if signals else 0


def check_anti_signals(response, anti_signals):
    r_lower = response.lower()
    return [s for s in anti_signals if s.lower() in r_lower]


# ── Main Evaluation ──────────────────────────────────────────
def run_eval(model, tokenizer, label):
    results = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "system_prompt": SYSTEM_PROMPT,
        "num_questions": len(EVAL_SUITE),
        "tests": [],
    }

    print(f"\n{'='*60}")
    print(f"  Cross-Scale TI Evaluation: {label}")
    print(f"  {len(EVAL_SUITE)} questions")
    print(f"{'='*60}\n")

    for i, test in enumerate(EVAL_SUITE):
        print(f"[{i+1}/{len(EVAL_SUITE)}] {test['name']}...")

        sys_prompt = test.get("system", SYSTEM_PROMPT)
        response, tokens, elapsed = generate(model, tokenizer, test["prompt"], system=sys_prompt)

        found_signals, signal_score = signal_check(response, test.get("signals", []))

        result = {
            "id": test["id"],
            "name": test["name"],
            "category": test["category"],
            "knowledge_sub": test.get("knowledge_sub"),
            "what_it_tests": test["what_it_tests"],
            "prompt": test["prompt"],
            "system_used": sys_prompt[:60] + "...",
            "response": response,
            "word_count": len(response.split()),
            "tokens_generated": tokens,
            "time_seconds": round(elapsed, 1),
            "signal_score": round(signal_score, 2),
            "signals_found": found_signals,
            "signals_expected": test.get("signals", []),
        }

        # Brevity check
        if "max_words" in test:
            result["brevity_target"] = test["max_words"]
            result["brevity_pass"] = result["word_count"] <= test["max_words"]

        # Anti-signal check
        if "anti_signals" in test:
            anti_found = check_anti_signals(response, test["anti_signals"])
            result["anti_signals_found"] = anti_found
            result["switching_pass"] = len(anti_found) == 0

        # Think tag check
        result["has_think_tag"] = "<think>" in response

        print(f"  Signals: {len(found_signals)}/{len(test.get('signals', []))} "
              f"({signal_score:.0%}) | Words: {result['word_count']} | "
              f"Time: {elapsed:.1f}s")
        print(f"  Preview: {response[:120]}...")
        print()

        results["tests"].append(result)

    # ── Summary by category ──
    category_scores = {}
    for t in results["tests"]:
        cat = t["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        if t["signals_expected"]:
            category_scores[cat].append(t["signal_score"])

    # ── Summary by knowledge subcategory ──
    knowledge_scores = {}
    for t in results["tests"]:
        ks = t.get("knowledge_sub")
        if ks and t["signals_expected"]:
            if ks not in knowledge_scores:
                knowledge_scores[ks] = []
            knowledge_scores[ks].append(t["signal_score"])

    scored = [t for t in results["tests"] if t["signals_expected"]]
    avg_signal = sum(t["signal_score"] for t in scored) / len(scored) if scored else 0
    avg_words = sum(t["word_count"] for t in results["tests"]) / len(results["tests"])

    results["summary"] = {
        "avg_signal_score": round(avg_signal, 2),
        "avg_word_count": round(avg_words),
        "total_time_seconds": round(sum(t["time_seconds"] for t in results["tests"]), 1),
        "think_tag_rate": round(
            sum(1 for t in results["tests"] if t["has_think_tag"]) / len(results["tests"]), 2
        ),
        "category_scores": {
            cat: round(sum(scores) / len(scores), 2) if scores else None
            for cat, scores in category_scores.items()
        },
        "knowledge_subcategory_scores": {
            ks: round(sum(scores) / len(scores), 2)
            for ks, scores in knowledge_scores.items()
        },
    }

    # Print summary
    print(f"{'='*60}")
    print(f"  SUMMARY — {label}")
    print(f"  Overall signal score: {avg_signal:.0%}")
    print(f"  Avg response length:  {avg_words:.0f} words")
    print(f"\n  By Knowledge Subcategory:")
    ks_names = {
        "A": "Labor Market Interpretation",
        "B": "Compensation & Benefits",
        "C": "Competitive Intelligence",
        "D": "Data Quality & Method Judgment",
        "E": "Strategic Workforce Planning",
        "F": "Stakeholder Comm & Commercial",
        "G": "Operational Intelligence Design",
    }
    for ks in sorted(knowledge_scores.keys()):
        scores = knowledge_scores[ks]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"    {ks}: {ks_names.get(ks, ks):40s} {avg:.0%} ({len(scores)} questions)")

    print(f"\n  By Behavioral Category:")
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"    {cat:30s} {avg:.0%} ({len(scores)} questions)")

    print(f"{'='*60}")

    return results


def main():
    p = argparse.ArgumentParser(description="Cross-Scale TI Evaluation")
    p.add_argument("--model", type=str, help="Model path (merged or base)")
    p.add_argument("--adapter", type=str, help="LoRA adapter path")
    p.add_argument("--base", type=str, help="Base model ID")
    p.add_argument("--label", type=str, required=True, help="Label for this run")
    p.add_argument("--output", type=str, help="Output JSON path")
    args = p.parse_args()

    if not args.model and not args.adapter:
        print("Provide --model or --adapter + --base")
        p.print_help()
        sys.exit(1)

    output_path = args.output or f"results/eval_{args.label}.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"\n  Loading model for evaluation: {args.label}")
    if args.adapter:
        model, tokenizer = load_model(args.adapter, args.base)
    else:
        model, tokenizer = load_model(args.model)

    results = run_eval(model, tokenizer, args.label)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
