# Tech Valley — code

Runnable code for the [Tech Valley](https://techvalley.substack.com) newsletter. Every number and figure in a post comes from the script in its folder.

| Issue | Folder |
|---|---|
| 1. The Variance Tax: Why Your A/B Tests Take Twice as Long as They Should | [issue-01-variance-reduction](issue-01-variance-reduction/) |

## Run

Requires Python 3 with numpy, scipy and matplotlib.

```
python issue-01-variance-reduction/analysis.py
```

All simulations are seeded, so the output matches the post. Figures are written to a `figures/` folder next to the script.

The A/B testing toolkit used in the posts lives at [claude-experimentation](https://github.com/LeihuaYe/claude-experimentation).
