# Synthetic AG News generation and quality analysis

## Dataset and purpose

AG News is a four-class news classification dataset with `text` and `label` fields. Its categories are World, Sports, Business, and Sci/Tech. This project uses the Hugging Face `fancyzhx/ag_news` copy. From the original training split, [`src/prepare_data.py`](../src/prepare_data.py) selects a class-balanced 8,000-article real training set (2,000 per class) and a separate 2,000-article validation set (500 per class), using seed 42. The original 7,600-article test set has 1,900 articles per class and is kept unchanged. The real articles are often short and may contain agency names, bylines, HTML entities, or other source artifacts. The generated articles are fictional headline-and-summary briefs; this format difference matters when comparing the two sources.

The synthetic dataset was created to test whether replacing part of the real training set changes DistilBERT classification performance. The final source contains 8,000 articles, again 2,000 per category. It is not a substitute for the real validation or test data.

## Generator and reproducibility

Generation used a locally hosted **Qwen3-4B-Instruct** model through Ollama (`qwen3:4b-instruct`, Q4_K_M variant, Ollama 0.40.11). The production generation metadata records temperature 0.85, `top_p=0.8`, `top_k=20`, and a 180-token output limit. Each requested article supplies an AG News category, a prompt, and a system instruction asking for an original, plausible, fictional English news brief. The requested response format is `Headline:` and `Summary:`. The generation code parses these fields, joins them as the article text, and stores a sample ID, target label, model digest, prompt hash, random seed, generation settings, and, for advanced prompts, scenario metadata. These records make it possible to trace generated rows to their source settings. See [`src/generate.py`](../src/generate.py) and the saved source [`full_synthetic.csv`](../data/synthetic/full_synthetic.csv).

The full initial generation comprised 8,000 rows: 200 basic and 1,800 advanced articles per category (800 basic and 7,200 advanced overall). The pilot and prompt-comparison exercises used smaller, separate samples; their statistics should not be interpreted as measurements on every final article.

## Prompting strategies

The [basic prompt](../prompts/basic.txt) requests one short article in a specified category, a headline, a two- or three-sentence summary, 35–80 words in total, and the required field format. The [advanced prompt](../prompts/advanced.txt) adds a concrete scenario, asks for specific actors and a news angle, discourages stock wording, and asks for a plausible fictional story with a clear primary category. Scenario topics, places, and angles are varied by the generation script. Both strategies use the same underlying model and production decoding settings.

A controlled prompt comparison selected 800 existing basic and 800 existing advanced articles, with 200 per class in each set. Advanced articles had higher first-35-word distinct-2 in all four categories: 0.602–0.662 versus 0.182–0.420 for basic articles. Their mean within-category sentence-embedding diversity, defined as one minus mean pairwise cosine similarity, was about 0.771 versus 0.515 for basic articles. Advanced articles were also longer on average (69.2 versus 57.2 words), which can contribute to diversity differences despite the length-matched lexical check. The separate 40-item human pilot scored category fit at 5.00/5 for advanced and 4.95/5 for basic, but realism at 2.9/5 and 3.0/5 respectively. Greater measured diversity did **not** establish greater realism. These results and their sampling method are in [`10_prompt_comparison.ipynb`](../notebooks/10_prompt_comparison.ipynb) and [`results/task006/comparison.csv`](../results/task006/comparison.csv).

## Quality screening and finalization

The quality pipeline in [`src/filter_data.py`](../src/filter_data.py) checks required fields, labels, text length, exact duplicates, near duplicates, and possible category mismatches. Near-duplicate screening uses Ollama `all-minilm` embeddings and a cosine threshold of 0.95. A label-screening step combines an AG News-trained embedding classifier with Qwen3-4B review of disagreements. Human and rule-based reviews flag selected realism, geographic, sports, or technical inconsistencies. A flagged model disagreement is a **candidate** for review, not proof of a wrong label; the automated checks do not verify every factual claim.

| Stage | Articles | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|---:|
| Initial full generation | 8,000 | 2,000 | 2,000 | 2,000 | 2,000 |
| After filtering | 7,427 | 1,824 | 1,827 | 1,890 | 1,886 |
| Final after screened replacements | 8,000 | 2,000 | 2,000 | 2,000 | 2,000 |

The filtering log attributes 234 removals to near duplicates, 223 to possible label mismatches, 36 to user realism review, and 80 to geographic or technical review. These counts describe pipeline decisions, not independently confirmed error rates. The filtered set is slightly class-imbalanced because exclusion rates differ across categories. Screened replacement articles restore 2,000 rows per class, yielding [`final_8000.csv`](../data/synthetic/final_8000.csv). That final source contains 7,402 advanced and 598 basic articles. The full stage counts, model identifiers, thresholds, and replacement audit are in [`results/full_filtering_stats.json`](../results/full_filtering_stats.json) and [`07_dataset_audit.ipynb`](../notebooks/07_dataset_audit.ipynb).

## Lexical and semantic diversity

The final diversity audit compares **100 real and 100 final synthetic articles per class**. It uses the first 35 words for lexical and topic comparisons, reducing the effect of the synthetic articles' greater length, and full text for embeddings. Distinct-2 measures the share of unique adjacent word pairs; embedding diversity is one minus mean within-class pairwise cosine similarity. These are sample-based measures, not estimates of factual quality.

| Class | Real distinct-2 | Synthetic distinct-2 | Real embedding diversity | Synthetic embedding diversity |
|---|---:|---:|---:|---:|
| World | 0.857 | 0.758 | 0.912 | 0.807 |
| Sports | 0.856 | 0.729 | 0.872 | 0.772 |
| Business | 0.882 | 0.755 | 0.900 | 0.774 |
| Sci/Tech | 0.889 | 0.719 | 0.897 | 0.772 |

Across the four class-level values, mean distinct-2 is about 0.871 for real and 0.740 for synthetic articles; mean embedding diversity is 0.896 versus 0.781. The final synthetic set has no exact duplicate full texts, but **208 articles share one of 68 repeated headlines**. Repeated openings include phrases such as “Quantum AI Team” in Sci/Tech and “NovaCore Announces Expansion” in Business. Full-text uniqueness therefore does not guarantee varied headlines or news angles. See [`11_diversity_bias.ipynb`](../notebooks/11_diversity_bias.ipynb).

## Balance, topic drift, and artifacts

Class balance is exact in the initial and final synthetic sets. The intermediate filtered set has a small deviation from equal class shares: total variation distance 0.0084. Balancing the final set prevents class-frequency differences from trivially driving the main training comparison, but it does not remove differences in wording or subject matter.

The matched 100-per-class audit shows that synthetic articles mention “AI” in 36% of Sci/Tech samples versus 0% of the sampled real articles; “AI powered” appears in 21% versus 0%. In Sports, “match” appears in 32% of synthetic versus 5% of real samples. Conversely, sampled real Sci/Tech articles mention “Microsoft” in 15% versus 0% synthetic, and real World articles mention “Iraq” in 12% versus 0% synthetic. These are **sample-specific topic and vocabulary differences**, not evidence of demographic bias. The real news source and fictional scenario prompts cover different times and event types, so several causes are possible.

Generation artifacts include reused headline templates, repeated fictional organization names, and unusually formulaic openings even when article bodies differ. In the small realism review, one basic Sports article contradicted known NBA history, and an advanced World article confused the Mont Blanc road tunnel with a rail corridor. Such cases illustrate why format validity and category fit cannot stand in for factual plausibility. The audit identifies concrete examples by sample ID in [`11_diversity_bias.ipynb`](../notebooks/11_diversity_bias.ipynb) and realism notes in [`10_prompt_comparison.ipynb`](../notebooks/10_prompt_comparison.ipynb).

## Limitations and interpretation

Automated filtering depends on the selected embedding model, cosine threshold, and label-screening model; it can miss implausible stories or remove useful variation. The human realism review covers only 40 pilot articles, and the real-versus-synthetic diversity comparison covers 400 articles per source, so neither validates all 8,000 final synthetic rows. Prompt strategies have unequal source-pool sizes and different average lengths. The final balanced dataset is mostly advanced-prompt text, which limits generalization about the basic strategy.

The downstream [training-results report](training_results.md) provides an external check on usefulness: 25% synthetic training matched or slightly exceeded the controlled real-only model on the fixed real test set, whereas 100% synthetic training substantially reduced macro F1 and Sci/Tech recall. This does not identify a single causal defect in the generated text. The quality evidence instead shows a balanced, usable but less diverse synthetic source with detectable topic and template artifacts, and a need for broader human review before stronger claims about realism or bias.
