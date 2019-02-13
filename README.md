# Named entity recognition for literary texts.
`figur` is a simple framework for recognizing named entities in German literary texts. The high-level API makes it _very_ easy to use:

```python
>>> import figur
>>> text = "Der Gärtner entfernte sich eilig, und Eduard folgte bald."
>>> figur.tag(text)
```

## Installation

```
$ pip install figur
```

## Getting started
The high-level API provides three entry points:
- `figur.train()` to train a new model.
- `figur.optimize()` to optimize hyperparameters.
- `figur.tag()` to tag tokens in a string.
