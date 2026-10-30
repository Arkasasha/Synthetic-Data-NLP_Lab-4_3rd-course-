# Speaker notes

Planned duration: 11 minutes 40 seconds. These timings include handoffs.

## Slide 1: Synthetic news for AG News

Oleksandr | 0:35 | Introduce the question and two linked workstreams: generating usable examples and testing them on real news.

## Slide 2: Research question

Oleksandr | 0:55 | Explain replacement rather than adding more data. State that the outcome is performance on original real articles.

## Slide 3: Dataset and split design

Oleksandr | 0:55 | Describe class balance and why keeping validation/test real makes this a test of transfer to actual AG News.

## Slide 4: Generate and screen synthetic news

Maksym | 1:10 | Explain prompt contrast and screened replacements. State that automatic filters are quality proxies, not factual guarantees.

## Slide 5: Data quality findings

Maksym | 1:10 | Explain distinct-2 and semantic diversity briefly. Mention the small review sample and repeated headline artifacts.

## Slide 6: Controlled model setup

Oleksandr | 0:55 | Distinguish this one-epoch comparison from the separate five-epoch hyperparameter grid.

## Slide 7: Main experiment results

Oleksandr | 1:15 | Read the graph: tiny lead at 25%, decline after 50%, sharp drop at 100%. Emphasize one seed and exploratory test-set ranking.

## Slide 8: Which class suffers?

Oleksandr | 0:55 | Point to class recall, not only overall accuracy. Explain mixed business/technology vocabulary as a possible cause.

## Slide 9: 50% ablation

Oleksandr | 1:05 | Clarify basic-enriched is 400 basic plus 3,600 advanced, not 100% basic. Do not claim filtering is harmful from these small differences.

## Slide 10: Paired error analysis

Oleksandr | 1:05 | Give one category-boundary example: a technology company story can be Business or Sci/Tech depending on its main angle.

## Slide 11: Discussion and conclusion

Maksym | 1:00 | Summarize the evidence without claiming a robust 25% optimum or a proven cause for every error.

## Slide 12: Contributions and sources

Both | 0:40 | Each presenter closes with their contribution. Refer to the report for full citations and caveats. Total planned speaking time: 11 min 40 sec.
