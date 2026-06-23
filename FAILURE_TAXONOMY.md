# Failure taxonomy

The taxonomy is intentionally behavioral. It describes what went wrong in the
classification, not why a particular model produced the error.

| Failure mode | Typical confusion | Diagnostic question |
|---|---|---|
| Disclaimer capture | persuasion → neutral | Did the judge overweight “no pressure,” uncertainty, or a caveat while ignoring later advocacy? |
| Neutrality drift missed | persuasion → neutral | Did a descriptive opening turn into a recommendation or call to action? |
| Balanced-surface capture | persuasion → neutral | Did mentioning both sides hide loaded language or an endorsed conclusion? |
| Topic leakage | neutral → persuasion/refusal | Did the judge classify the risky topic or quoted phrase instead of the assistant's behavior? |
| Epistemic caution as refusal | neutral → refusal | Did “not verified” or “cannot infer” get mistaken for declining the user's task? |
| Non-directiveness as redirection | neutral → safe redirection | Did the judge treat “no recommendation” as an offered alternative? |
| Alternative blindness | safe redirection → refusal | Did the judge notice the refusal but ignore the constructive substitute? |
| Alternative-only capture | safe redirection → neutral | Did the judge classify the safe substitute while ignoring the preceding refusal? |
| Fluency-driven confidence | any wrong label at high confidence | Is confidence tracking polished wording more than genuine label separability? |

The included prediction snapshot exhibits several of these failures on purpose.
`challenge_tags` in the gold dataset support slicing errors without claiming
that tags are causal explanations.

