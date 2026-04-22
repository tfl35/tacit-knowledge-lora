# A Practitioner's Guide to Encoding Domain Expertise

*Lessons from Cross-Scale Fine-Tuning of Talent Intelligence Reasoning*

This document distills the practical methodology lessons from the research into guidance for any domain expert who wants to encode their analytical reasoning into an open-weight language model through parameter-efficient fine-tuning. The lessons are sequenced in the order a practitioner would encounter them: dataset construction, training configuration, evaluation design, deployment, and governance. Each lesson was discovered through iterative experimentation, and none could have been anticipated from static planning alone.

---

## Lesson 1: Capture What Experts Do, Not What They Know

The training dataset is not a knowledge base. It is a collection of behavioral demonstrations. The distinction matters because it determines what you write.

A knowledge base entry might read: "When evaluating compensation surveys, check for sample size, industry representation, and geographic relevance." A behavioral demonstration places those same considerations inside a stakeholder scenario where the analyst is responding to a real question, pushing back on a flawed premise, or asking a diagnostic question before answering. The gradient descent process does not need an expert to explain why they evaluate methodology before trusting data. It needs to see them doing it, across enough contexts, for the pattern to emerge.

This framing connects to the project's most significant theoretical finding: that at least some dimensions of tacit expertise — specifically, behavioral patterns like diagnostic questioning — transferred more readily through behavioral demonstration to a pattern learner than through structured articulation to a human learner. The gradient descent process bypasses the articulation bottleneck: it does not need the expert to explain why, only to demonstrate what. The practical implication is that expert time should be spent reviewing and refining scenario-response pairs, not writing process documentation.

## Lesson 2: The Sweet Spot Is Smaller Than You Think

The ablation study established that 100 to 150 well-constructed examples produce meaningful domain transfer on a 9B-parameter model with full general knowledge preservation. Below 100, domain encoding is present but response quality is less calibrated. Above 250, general knowledge begins to erode.

The practical translation: one domain expert working from professional experience can direct the generation, review, and refinement of 125 examples in two to three working days. Those examples, trained for three epochs, produce a model that scores at Level 3 to 4 analytical maturity on qualitative evaluation.

The organizational question is no longer whether this is feasible. It is whether the examples are good enough.

## Lesson 3: Never Combine Datasets with Different System Prompts

This is the single most destructive mistake made during the project, and the one most likely to be repeated by practitioners who assume that more data is always better.

When the 252-example talent intelligence dataset was combined with a 403-example general-purpose reasoning dataset that used a different system prompt, the resulting model produced template fragments, hallucinated multi-turn continuations, and lost all domain knowledge. The competing system prompts created catastrophic interference. The recovery required undoing the combination entirely and returning to the lightest possible intervention on the original dataset.

The generalized lesson: training data should use a single normalized system prompt. If you need to vary the model's behavior across contexts, do so through inference-time system prompting, not through training data heterogeneity. Knowledge encoding and output formatting are separable concerns. Fine-tuning controls what the model knows and how it reasons. System prompting controls how it presents its output.

## Lesson 4: The Structure of Your Examples Becomes the Structure of Your Outputs

If 90% of your training responses build context for twelve paragraphs before delivering the actionable insight in the final paragraph, your model will bury the lead on every question. If all your responses cluster at 300 to 400 words with no short responses, your model will produce uniform-length answers regardless of question complexity. If only 10% of your responses open with a recommendation, your model will almost never lead with one.

This was confirmed quantitatively: the structural distribution of the training data predicted the model's output patterns with high fidelity.

The fix is to audit the structural distribution before training. Count how many responses lead with insight versus build-up. Count the distribution of response lengths. Count the proportion of single-turn versus multi-turn examples. If the distribution does not match how you want the model to behave, reshape it before training. Reshaping after the fact — by adding formatting layers on top of existing content — was more destructive than the original problem.

## Lesson 5: Dataset Size and Epoch Count Interact Non-Linearly

At 252 training examples, five epochs produced stable training with no quality degradation. At 350 examples, five epochs caused response looping, cached responses to unrelated prompts, and hallucinated conversation continuations past the intended stop point. Reduction to three epochs eliminated all artifacts while preserving domain knowledge transfer.

This interaction is counterintuitive: more training data typically permits more training passes. Practitioners should re-validate epoch count whenever the dataset changes significantly, starting at three epochs and increasing only if evaluation confirms that the model has not yet converged.

## Lesson 6: Build at Least Two Evaluation Methods That Cannot Agree by Construction

The most important methodological lesson from this project is that a single evaluation method will mislead you, and you will not know it is misleading you until a second method contradicts it.

Signal-based keyword scoring indicated that the 9B fine-tuned model had regressed from its base state. LLM-as-judge evaluation scored it at 3.46/5, higher than any other model. The signal–judge correlation was *r* = .08 — essentially noise. Had only signal scoring been used, the project's best results would have been interpreted as failures.

The reason is structural: as domain knowledge transfer deepens, the model paraphrases rather than reproducing expected vocabulary, making keyword evaluation worse at precisely the moment the model is getting better. This creates a practical paradox: the models most in need of qualitative evaluation are those whose quality is hardest to assess automatically.

Build at least one automated method and one qualitative method. When they disagree, investigate the disagreement rather than averaging the scores. The disagreement is often the most informative finding.

## Lesson 7: Start with the 4B, Not the Largest Model You Can Run

The 4B model achieved the highest signal density ratio (2.4×), the best compression efficiency, and scored 3.18/5 on qualitative evaluation — Level 3 analytical maturity on Culshaw's (2022) framework.

The 9B scored higher on reasoning strategy and epistemic calibration but lower on actionability, and exhibited system prompt switching instability that the 4B did not. For deployment to the organizations this project targets — community workforce programs, small nonprofits, local governments — the 4B is the right starting point. It runs on any laptop with 8 GB RAM, produces a response in 10 to 30 seconds on CPU, and costs nothing to operate.

The 9B is better for experienced users who can evaluate its output critically. Start where your users are, not where the benchmark is highest.

## Lesson 8: Document the Adapter, Not Just the Model

A fine-tuned adapter encodes one expert's analytical priorities, emphasis patterns, and blind spots. The vocabulary shift analysis quantified this directly: the model amplified compensation terms by 1.87× and labor market terms by 1.54× while suppressing data quality terms to 0.45×, reflecting the training data's emphasis distribution. Anyone who downloads and uses the adapter inherits those priorities without knowing they exist — unless the documentation explains them.

A detailed model card covering training provenance, evaluation results, known emphasis biases, and responsible use framing is not optional. It is the difference between a reasoning aid and an oracle that users trust without calibration. The model card should state what expert's judgment the adapter encodes, what domains are strongest and weakest, what the evaluation found and did not find, and that the model should function as a thought partner rather than an authoritative source.

See [`model_card.md`](model_card.md) for the template used in this project.

---

## Applying This to Your Own Domain

These eight lessons describe a replicable methodology for encoding domain expertise through parameter-efficient fine-tuning. The talent intelligence application was the first test case. The process generalizes to any field where practitioner judgment is the valuable layer above accessible data.

The practitioners best positioned to learn this are those who already teach, mentor, or onboard others in their domain — because they have already practiced the translation from tacit judgment to structured demonstration. This methodology formalizes that translation into a repeatable, measurable process.

The organizations best positioned to use it are not the ones with the most data scientists. They are the ones with the clearest understanding of what expertise they possess, what behavioral examples best represent it, and what decisions would improve if that expertise were more widely accessible.

## References

Culshaw, T. (2022). *Talent intelligence: Use business and people data to drive organizational performance.* Kogan Page.

Nonaka, I., & Takeuchi, H. (1995). *The knowledge-creating company: How Japanese companies create the dynamics of innovation.* Oxford University Press.
