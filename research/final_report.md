# Encoding Talent Intelligence Expertise into Open-Weight Language Models

## A Cross-Scale Knowledge Fidelity Analysis

---

## Introduction

Talent intelligence is a judgment-intensive analytical discipline whose value resides in interpretation rather than data access (Culshaw, 2022). Experienced practitioners evaluate labor market signals, challenge flawed premises, synthesize competitive intelligence from disparate sources, and deliver recommendations calibrated to stakeholder context. This expertise is concentrated in large enterprises with dedicated teams; community workforce programs, small nonprofits, and local government agencies operate without it despite arguably needing it most.

This project asks a question at the intersection of three domains: when a domain expert's tacit analytical reasoning is structured into training examples and encoded into model weights through parameter-efficient fine-tuning, which dimensions of that expertise survive, and which resist? The question matters because each contributing literature contains an assumption the others can test. Talent intelligence practitioners assume their judgment is irreducibly human. Knowledge management theorists assume that externalization processes produce predictable losses. Machine learning researchers assume that fine-tuning encodes task performance without asking whether the kind of knowledge matters. This project sits at the intersection and provides empirical evidence none of these literatures offers on its own.

The project fine-tunes the Qwen3.5 model family across four parameter scales (9B, 4B, 2B, and 0.8B) on 350 talent intelligence examples using bf16 LoRA (Hu et al., 2021), then evaluates which of seven knowledge subcategories transfer most faithfully at each scale using eight complementary evaluation methods, including a dataset size ablation study and a logit-level behavioral divergence analysis adapted from Zhu et al. (2025).

**The project makes five contributions.** First, a knowledge fidelity matrix demonstrating that fine-tuning compresses rather than degrades domain reasoning, with 1.6× to 3.2× signal density improvement across scales. Second, a methodological finding that automated keyword evaluation becomes unreliable as knowledge transfer deepens, with the 9B model's signal–judge correlation at *r* = .08. Third, a dataset optimization finding that 100 to 150 examples represents the practical sweet spot for adapter training. Fourth, a mechanistic finding from logit-level divergence analysis that fine-tuning restructured generative priors near-completely (2.1% token agreement). Fifth, a practical finding that the 4B model delivers Culshaw's (2022) Level 3 talent intelligence reasoning on consumer hardware, free and offline.

## Literature Review

### Talent Intelligence as an Intelligence Discipline

Culshaw (2022) provides the most comprehensive treatment of talent intelligence, positioning it within the broader taxonomy of intelligence disciplines and drawing explicit parallels to SIGINT, GEOINT, HUMINT, and OSINT from national security contexts. His HUMINT framework maps onto the diagnostic questioning pattern central to this project's training data: the instinct to ask scoping questions of a stakeholder before analyzing their data. His Data Confidence Model, adapted from UK police intelligence reporting, classifies intelligence across Document Confidence, Data Source Classification, and Data Point Classification dimensions, providing a formal epistemology that maps directly onto this project's epistemic calibration evaluation dimension.

Culshaw's five-level maturity model (Experimental through Transformational) provides a framework for calibrating what fine-tuning can realistically achieve. The training data aims to encode Level 3 to 4 (Operational to Strategic) reasoning. Level 5 capabilities — relationship-dependent trust and senior leadership navigation — likely resist parametric encoding, consistent with Nonaka and Takeuchi's (1995) predictions. Bersin (2024) complements Culshaw by documenting talent intelligence's emergence as a technology category, noting that enterprise platforms automate data aggregation while leaving the interpretive layer unaddressed.

### Tacit Knowledge and the Limits of Externalization

Nonaka and Takeuchi (1995) established the foundational SECI framework for understanding how tacit knowledge is converted into explicit, transferable forms. Their externalization phase describes surfacing tacit expertise through dialogue, metaphor, and structured reflection. This project's training data generation maps onto that phase: the researcher replays stakeholder scenarios, structures analytical reasoning into prompt-response pairs, and constructs contrastive examples that surface judgment patterns. Critically, Nonaka and Takeuchi observed that pattern recognition, intuitive calibration, and "negative knowledge" — knowing what to ignore — resist structured formalization.

However, their framework was developed for organizational knowledge creation through human dialogue, not for machine learning pipelines. The externalization medium here is fundamentally different: gradient descent does not need the expert to explain why they ask a scoping question before answering — it only needs to see that they do, in enough contexts, often enough to learn the shape of when. Whether Nonaka and Takeuchi's predictions hold when the recipient is a neural network rather than a human learner is an open empirical question, and arguably the most consequential one this project addresses.

### Parameter-Efficient Fine-Tuning

Hu et al. (2021) introduced Low-Rank Adaptation (LoRA), demonstrating that weight updates during task adaptation reside on a low intrinsic dimension, providing mathematical grounding for the claim that a small number of well-constructed expert examples can produce meaningful behavioral changes. Dettmers et al. (2023) extended LoRA with QLoRA for accessible hardware scales. Losch et al. (2025) demonstrated meaningful performance gains from as few as 200 to 300 medical training examples, though clinical classification is a bounded task with clear correct answers, while talent intelligence reasoning is open-ended and judgment-dependent.

### Behavioral Divergence and the Identified Gap

Zhu et al. (2025) introduced a mechanistic analysis framework for studying token-level behavioral divergence, which this project adapts to compare base and fine-tuned model outputs through agreement rate, entropy delta, and domain token rank shifts. The literature establishes that talent intelligence is a judgment-intensive discipline (Culshaw, 2022), that tacit expertise externalizes with predictable losses (Nonaka & Takeuchi, 1995), and that parameter-efficient fine-tuning makes encoding such knowledge feasible (Hu et al., 2021; Dettmers et al., 2023). What is missing is empirical evidence at the intersection: no study has encoded a domain expert's tacit analytical reasoning via fine-tuning and systematically measured which dimensions survive across model scales and dataset sizes using multi-method evaluation with mechanistic analysis. This project fills that gap.

## Methodology

### Dataset Construction

The training dataset follows a three-phase construction methodology. AI-assisted synthetic generation (Anthropic's Claude) provided a foundation, with the researcher's domain expertise applied through iterative guidance, gap analysis, and manual review. In Phase 1, the researcher directed generation of 150 base examples by specifying analytical trajectories, stakeholder personas, and reasoning patterns drawn from professional talent intelligence experience. A three-pass revision process smoothed vendor de-identification artifacts, rewrote technical language into decision-level framing, and converted 18 single-turn examples into multi-turn diagnostic sequences. In Phase 2, deep analysis of Culshaw (2022) generated 102 additional examples across eight thematic batches addressing gaps in analytical frameworks, case study patterns, and operationalized intelligence scenarios.

Phase 3 used iterative evaluation-driven refinement across four training-evaluation cycles, producing several methodological findings. The most critical was that combining datasets with different system prompts caused catastrophic interference: when a 403-example general-purpose dataset was merged with the talent intelligence dataset, the model produced template fragments rather than analytical reasoning. This led to a key insight: knowledge encoding and output formatting are separable concerns. Fine-tuning encodes what and how the model reasons; inference-time system prompting controls presentation format. A third finding was that the distribution of response structures directly shapes model behavior: when 90% of training responses built context before delivering insight, the model learned to bury the lead.

The interaction between dataset size and epoch count proved non-linear: 252 examples tolerated five epochs, but 350 examples produced response looping, requiring reduction to three. Diagnostic questioning — the analyst's instinct to ask scoping questions before answering — resisted encoding even when 25% of training examples demonstrated the behavior, because the remaining 75% actively reinforced the opposite pattern. The final 350-example dataset contains 242 single-turn and 108 multi-turn conversations totaling approximately 185,000 words, with response lengths deliberately varied from 42 to 433 words.

### Experimental Design

The experiment trains the identical 350-example dataset on four Qwen3.5 model sizes (9B, 4B, 2B, and 0.8B) for three epochs each using bf16 LoRA with scaled hyperparameters. All four models share the same hybrid attention architecture — interleaved Gated DeltaNet and full softmax attention in a 3:1 ratio (Yang et al., 2025) — the same tokenizer (248,320 vocabulary), and the same chat template, isolating parameter capacity as the single independent variable. LoRA rank scales with model size (r = 64 for 9B/4B, r = 32 for 2B, r = 16 for 0.8B). Each model is evaluated in both base and fine-tuned states on a 45-question evaluation suite, producing eight evaluation runs. A dataset size ablation study trained separate 9B adapters on stratified subsets of 50 to 300 examples. A behavioral divergence study captured generation logits from the 9B base and fine-tuned models across 15 evaluation questions. The full cross-scale pipeline ran in 5.2 GPU-hours on an NVIDIA DGX Spark (128 GB unified memory).

### Evaluation Framework

The evaluation employs eight complementary methods: signal-based scoring with enhanced fuzzy matching; visible signal density (signals per 100 visible words) to normalize for verbosity; LLM-as-judge evaluation using Claude Sonnet 4.6 across five qualitative dimensions; vocabulary shift analysis; brevity calibration analysis; Culshaw maturity model mapping; dataset size ablation; and logit-level behavioral divergence analysis. Because this project evaluates open-ended analytical reasoning rather than classification, traditional metrics (accuracy, precision, recall, F1-score) do not apply.

## Results

### Cross-Scale Performance

Fine-tuning compressed domain reasoning by 1.6× to 3.2× across all scales as measured by visible signal density. The 4B model achieved the highest visible density (1.35 signals per 100 words) and strongest density improvement ratio (2.4×). The LLM-as-judge evaluation restored the expected scale ordering (9B: 3.46, 4B: 3.18, 2B: 2.45) while revealing a near-zero signal–judge correlation for the 9B (*r* = .08), confirming that automated keyword evaluation fails to capture deeply internalized domain reasoning. No catastrophic forgetting was detected at any scale.

**Table 1: Cross-Scale Performance Summary**

| Metric | 9B | 4B | 2B | 0.8B |
|---|---|---|---|---|
| Visible Signal Density | 1.02 | 1.35 | 1.03 | 0.48 |
| Density Ratio (FT/Base) | 1.7× | 2.4× | 3.2× | 1.6× |
| Avg Words (Base → FT) | 1,227 → 272 | 1,184 → 253 | 766 → 249 | 714 → 395 |
| Judge Score (1–5) | 3.46 | 3.18 | 2.45 | — |
| Signal–Judge *r* | .08 | .44 | .37 | — |
| Gen. Knowledge | 0.88 | 0.88 | 0.71 | 0.71 |

*Visible signal density = signals per 100 visible words. Judge scores unavailable for 0.8B due to response quality below evaluation threshold.*

### Knowledge Fidelity Matrix

Analysis across seven knowledge subcategories reveals that different dimensions of expertise transfer with different fidelity at different scales. Two subcategories survive to 0.8B without significant degradation: Labor Market Interpretation and Stakeholder Communication. Stakeholder Communication shows a counterintuitive pattern, actually increasing at 0.8B (0.249 vs. 0.227 at 9B), suggesting that interpersonal communication framing is the most scale-invariant dimension of talent intelligence. Three subcategories (Compensation, Competitive Intelligence, Strategic Workforce Planning) first degrade at 2B. Two (Data Quality, Operational Intelligence Design) degrade already at 4B.

**Table 2: Knowledge Fidelity Matrix (Fuzzy Signal Scores, Fine-Tuned Models)**

| Subcategory | 9B | 4B | 2B | 0.8B | Degrades at |
|---|---|---|---|---|---|
| A. Labor Market Interp. | 0.167 | 0.320 | 0.193 | 0.147 | Survives 0.8B |
| B. Compensation & Ben. | 0.405 | 0.452 | 0.143 | 0.198 | 2B |
| C. Competitive Intel | 0.202 | 0.171 | 0.146 | 0.098 | 2B |
| D. Data Quality & Meth. | 0.260 | 0.146 | 0.140 | 0.153 | 4B |
| E. Strategic WF Plan. | 0.307 | 0.337 | 0.190 | 0.075 | 2B |
| F. Stakeholder Comm. | 0.227 | 0.216 | 0.223 | 0.249 | Survives 0.8B ↑ |
| G. Ops Intel Design | 0.396 | 0.216 | 0.317 | 0.136 | 4B |

*Degradation defined as >20% drop from 9B score.*

### The Diagnostic Questioning Reversal

Signal-based scoring indicated that diagnostic questioning failed to transfer at every scale, appearing to confirm Nonaka and Takeuchi's (1995) prediction about negative knowledge resisting externalization. The LLM-as-judge evaluation reversed this finding: the 9B scored 4.0/5 on diagnostic questioning, with reasoning strategy at 4.75/5 and epistemic calibration at 4.75/5. The behavior transferred successfully but manifests as contextually appropriate scoping questions rather than detectable vocabulary patterns. This is the project's most consequential single finding: certain dimensions of tacit expertise resist automated detection even after successful encoding.

### The 9B Paradox

The 9B model's six zero-signal responses averaged 3.27/5 from the judge. The highest-rated response (4.6/5, with 5/5 on reasoning, depth, and epistemic calibration) scored 0.0 on signal matching. Vocabulary shift analysis confirmed the mechanism: analytical verbs dropped to 0.24× their base frequency because the model stopped announcing analysis and started performing it. Compensation terms rose 1.36× and labor market terms rose 1.50×, showing selective domain vocabulary amplification alongside analytical verb suppression. The model that most deeply internalized the domain produced the shortest, densest responses — a compression pattern that every practitioner who has worked alongside genuine experts will recognize.

### Dataset Size Ablation

The ablation study treats dataset size as a continuous hyperparameter on the 9B model. Signal score declines monotonically (0.392 at 50 examples to 0.259 at 300), but this is a word-count artifact: response length contracts from 853 to 243 words while signal density rises throughout (0.46 to 0.78), confirming compression rather than degradation. A critical threshold emerges at 250 examples where general knowledge preservation drops from 1.0 to 0.50. The practical sweet spot of 100 to 150 examples balances strong domain signal density, appropriate response length, and full general knowledge preservation.

**Table 3: Ablation Study — Overall Metrics by Dataset Size (9B Model)**

| Metric | 50 | 100 | 150 | 200 | 250 | 300 | Prod. |
|---|---|---|---|---|---|---|---|
| Signal score | 0.392 | 0.357 | 0.305 | 0.286 | 0.278 | 0.259 | 0.297 |
| Signal density | 0.46 | 0.57 | 0.58 | 0.66 | 0.68 | 0.78 | 1.02 |
| Avg words | 853 | 620 | 470 | 356 | 298 | 243 | 272 |
| Gen. knowledge | 0.875 | 1.0 | 1.0 | 1.0 | 0.50 | 0.835 | 0.88 |

*Signal density = signals per 100 visible words. Production result from cross-scale pipeline with full 350-example dataset.*

### Behavioral Divergence

The behavioral divergence study completes the full Zhu et al. (2025) framework. The overall token agreement rate of 2.1% confirms near-complete restructuring of generative behavior. The overall entropy delta of +0.143 indicates the fine-tuned model is less certain than the base model — a counterintuitive finding that resolves as appropriate analytical uncertainty, mechanistically consistent with the judge's high epistemic calibration scores (3.88/5). Subcategory B (Compensation and Benefits) is the exception: 5.9% agreement rate and near-zero entropy delta (+0.005), constituting the strongest multi-method finding in the project. Domain token rank shift analysis reveals that fine-tuning demoted "competitor" by 10.3 rank positions while promoting "benchmark" by 10.0, providing the mechanistic explanation for Competitive Intelligence's persistent weakness.

**Table 4: Behavioral Divergence by Knowledge Subcategory (9B Base vs. 9B Fine-Tuned)**

| Subcategory | Agreement Rate | Entropy Delta | Interpretation |
|---|---|---|---|
| A. Labor Market | 0.4% | +0.130 | Near-total divergence |
| B. Compensation | 5.9% | +0.005 | Genuine internalization |
| C. Competitive Intel | 0.6% | +0.139 | Near-total divergence |
| D. Data Quality | 3.5% | +0.112 | Moderate divergence |
| E. Strategic WF Plan. | 1.5% | +0.177 | High uncertainty increase |
| F. Stakeholder Comm. | 0.1% | +0.220 | Minimal agreement |
| G. Ops Intel Design | 0.6% | +0.278 | Highest uncertainty increase |
| Overall | 2.1% | +0.143 | Near-complete restructuring |

*Positive entropy delta indicates increased uncertainty. Agreement rate = fraction of positions where base and fine-tuned models predict the same top-1 token.*

## Discussion

### Compression as the Signature of Expertise

The headline finding challenges a common assumption about fine-tuning small models: that domain knowledge is lost as parameters decrease. The signal density data shows the opposite. Fine-tuning teaches the model to deliver domain reasoning more efficiently, producing more analytical content per word. The ablation study extends this along the dataset-size axis: the same compression mechanism operates regardless of whether the model is getting smaller or the training set is growing. Verbosity-normalized metrics are essential when comparing models that produce structurally different outputs.

This compression pattern is more than a measurement observation. Every practitioner who has worked alongside genuine experts recognizes it: the senior analyst's answer to a complex question is shorter than the junior analyst's, more precise, and lands closer to the decision point. The junior analyst builds context for twelve paragraphs and then delivers a hedged recommendation. The senior analyst names the decision, the relevant variables, and the appropriate caveats in four sentences. The model that most deeply internalized the domain produced the shortest, densest responses — it stopped announcing analysis and started performing it. This finding provides empirical grounding for a critique that talent development practitioners have long observed informally: learning and development programs that measure expertise acquisition through articulation — can the learner explain the framework, cite the methodology, demonstrate the vocabulary — are tracking the wrong thing. They track the junior analyst stage, where knowledge is still external enough to be recited.

### Nonaka and Takeuchi Were Right About the Wrong Recipient

The diagnostic questioning reversal forces a more precise claim about the limits of tacit knowledge transfer. Nonaka and Takeuchi's (1995) predictions about externalization limits appear to hold when the recipient is a human learner receiving structured training materials. The 9B fine-tuned model achieved a judge score of 4.0/5 on diagnostic questioning despite signal-based scoring indicating total failure. The behavior encoded so deeply that it bypassed the vocabulary signature entirely and manifested as contextually appropriate analytical behavior. Gradient descent, operating on a static dataset with no ability to ask clarifying questions, encoded a behavioral pattern that human instructional design reliably fails to transfer: the instinct to ask before answering.

The theoretical implication requires careful framing: this is a single result in a single domain at a single model scale. It does not overturn Nonaka and Takeuchi's framework. But it suggests that some dimensions of tacit expertise — specifically, behavioral patterns like diagnostic questioning — may transfer more readily through behavioral demonstration to a pattern learner than through structured articulation to a human learner. The ceiling on externalization may depend on the medium more than the field has assumed. A training example does not need to articulate the reasoning behind a behavior; it only needs to demonstrate the behavior in sufficient context for pattern learning to occur. Whether this finding generalizes across domains, model architectures, and expertise types is an open empirical question that replication studies must address.

### The Evaluation Reliability Gradient

The signal–judge correlation gradient (*r* = .08 at 9B, *r* = .44 at 4B, *r* = .37 at 2B) describes a structural relationship with implications beyond machine learning: as any system more deeply internalizes a domain, the surface signals of that expertise become less reliable indicators of its presence. At shallow transfer, keyword matching is a reasonable proxy. At deep transfer, the model reasons in its own analytical voice and keyword matching becomes noise. The behavioral divergence study provides the mechanistic explanation: at 2.1% token agreement, the 9B's generative priors have been so thoroughly restructured that its vocabulary diverges from expected signal keywords even when its analytical reasoning is highest quality. This creates a practical paradox: the models most in need of qualitative evaluation are those whose quality is hardest to assess automatically.

### The Democratization Case and Practical Implications

The 4B model delivers Culshaw's (2022) Level 3 talent intelligence reasoning on a consumer laptop with 8 GB RAM, scoring 3.18/5 from the LLM judge and achieving the highest visible signal density. For organizations currently at Level 1 to 2, this represents a meaningful capability jump previously available only to enterprises with dedicated teams. The ablation finding that 100 to 150 examples is sufficient strengthens this argument: a domain expert with a half-day authoring session could construct a viable adapter. The organizational question is no longer whether this is possible; it is how to govern it.

The minimum viable adapter finding reframes the organizational capability problem. The historical bottleneck has been expert scarcity. This project suggests the bottleneck has shifted to the quality and representativeness of the examples those practitioners contribute. A strategic approach to organizational capability building now includes a deliberate process for capturing and encoding expert behavioral examples — not documenting what the expert knows, but documenting what they do, how they respond to ambiguity, and where they push back. That documentation is the training dataset.

### Limitations

Three limitations bound the claims this project can make. First, the evaluation measures whether the model reasons like an expert — whether its analytical framing, source skepticism, and epistemic calibration match expert patterns as assessed by qualitative judges and automated metrics. It does not measure whether following the model's recommendations improves actual workforce decisions. Downstream task evaluation with real labor market data and measurable decision outcomes is the critical gap between a validated analytical behavior and a validated decision-support system. Second, all findings derive from a single domain (talent intelligence), a single model architecture family (Qwen3.5), and a single expert's training data. The diagnostic questioning reversal and the compression-as-expertise-signature finding are suggestive but not yet established as general phenomena. Third, the LLM-as-judge evaluation, while essential for capturing quality that keyword metrics miss, introduces its own biases — the judge model's assessment of "expert-level reasoning" may not align with how domain practitioners would evaluate the same outputs.

### Challenges Encountered

Three significant challenges shaped the project. The dataset combination failure was the most consequential: merging a structurally sophisticated reasoning dataset with the talent intelligence dataset caused catastrophic quality degradation. The recovery required undoing nearly all structural modifications and returning to the lightest possible intervention. The lesson — that adding complexity to a training pipeline is almost always destructive — informed all subsequent design decisions. The 27B memory failure established a documented boundary condition for the hardware-architecture combination. The evaluation framework's initial inadequacy was the most intellectually productive challenge: discovering that the evaluation was systematically penalizing the best results forced the development of three additional evaluation methods and ultimately produced the project's methodological contribution.

## Future Work

Five extensions would meaningfully strengthen these findings. First, training the 27B model on hardware with sufficient memory would test whether Level 4 maturity is achievable through fine-tuning alone; the specific prediction, based on the observed gradient, is that the 27B would show lower entropy delta reflecting more confident internalization and higher actionability closing the gap with epistemic calibration. Second, replication in a different domain is already underway: the researcher is collaborating with an acupuncture practitioner to apply this methodology to a clinical reasoning domain, testing whether the encoding process generalizes when the researcher is curating another expert's knowledge rather than encoding their own. Third, multi-expert training data from two or more talent intelligence practitioners would test whether single-analyst bias can be mitigated while preserving analytical coherence. Fourth, a dedicated compensation adapter would test whether the most robustly transferable subdomain performs even better as a focused training objective. Fifth, downstream task evaluation presenting the model with actual workforce data would test whether the analytical frameworks transfer to operational decision-making contexts with messy, incomplete real-world inputs.

## Conclusion

This project demonstrates that a domain expert's tacit analytical reasoning can be meaningfully encoded into open-weight language model weights through parameter-efficient fine-tuning, with measurable behavioral shifts across all model scales and dataset sizes tested. The knowledge fidelity matrix shows that different dimensions of expertise transfer with different fidelity at different scales, and the finding that Stakeholder Communication survives to 0.8B with increasing performance — the most interpersonal dimension proving the most scale-invariant — was counterintuitive.

The diagnostic questioning reversal carries implications beyond this project. A behavior that knowledge management theory predicted would resist encoding did encode — so deeply that it left no surface trace. This is a single finding in a single domain, and replication across other expertise types is needed before drawing general conclusions. But it suggests that for behavioral patterns specifically, the ceiling on externalization may be higher when the recipient learns from demonstration rather than articulation. Fine-tuning offers a different medium, and the results observed here warrant further investigation.

The compression finding — that deeper expertise produces fewer words, not more — challenges how organizations measure and develop analytical capability. Every competency model, performance review rubric, and certification exam that rewards articulation over judgment is measuring the early stage of expertise development and calling it mastery. This project gives that critique empirical grounding it has not had before.

The practical contribution stands on its own: a 4B model delivering Level 3 talent intelligence reasoning, free, offline, on consumer hardware, buildable from approximately 125 expert examples in a half-day authoring session. The access gap described in this project's motivation — community workforce programs, small nonprofits, and local governments making labor market investments without analytical judgment — is now addressable in a way it was not before. Not solved, but addressable. The method is replicable, the models are open, and the methodology provides a template for encoding domain expertise in other fields. That is the contribution that is larger than the project itself.

## References

Bersin, J. (2024). Enterprise talent intelligence arrives, disrupting the HR tech market. *The Josh Bersin Company.* https://joshbersin.com/2024/05/enterprise-talent-intelligence-has-arrived-new-research-available/

Culshaw, T. (2022). *Talent intelligence: Use business and people data to drive organizational performance.* Kogan Page.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems, 36.* https://arxiv.org/abs/2305.14314

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685.* https://arxiv.org/abs/2106.09685

Losch, N., Plagwitz, L., & Büscher, A. (2025). Fine-tuning LLMs on small medical datasets: Text classification and normalization effectiveness on cardiology reports and discharge records. *arXiv preprint arXiv:2503.21349.* https://arxiv.org/abs/2503.21349

Nonaka, I., & Takeuchi, H. (1995). *The knowledge-creating company: How Japanese companies create the dynamics of innovation.* Oxford University Press.

World Economic Forum. (2025). *The future of jobs report 2025.* https://www.weforum.org/publications/the-future-of-jobs-report-2025/

Yang, S., Kautz, J., & Hatamizadeh, A. (2025). Gated delta networks: Improving Mamba2 with delta rule. *Proceedings of the Thirteenth International Conference on Learning Representations.* https://arxiv.org/abs/2412.06464

Zhu, R., Liu, Y., Sun, Z., Wang, Y., & Hu, W. (2025). When can large reasoning models save thinking? Mechanistic analysis of behavioral divergence in reasoning. *arXiv preprint arXiv:2505.15276.* https://arxiv.org/abs/2505.15276
