# Smart Reading & Writing Assistant — NLP Pipeline

A lightweight, implementation-focused NLP pipeline covering the five stages in the supplied university brief, plus the optional **+10 autocomplete-as-you-type** demo.

## Project structure

```text
spell_checking/          Stage 1: Levenshtein + context-aware correction + autocomplete
syntactic_processing/    Stage 2: spaCy POS/dependencies + hand-written CFG parser
semantic_analysis/      Stage 3: NER/semantic roles + Lesk-style WordNet WSD
discourse_pragmatics/   Stage 4: coreference + discourse connectives + pragmatic inference
tests/                  Regression and dedicated stage tests + 10 samples
pipeline.py             Stage 5: process(raw_text)
run.py                  CLI entry point
evaluate_samples.py     Processes all 10 samples and prints summary JSON
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
```

The spaCy model provides POS tags, dependency parses, and named entities. The project also contains a small fallback POS path so the hand-built CFG and lightweight demo remain usable when the model is not installed.

## Run the pipeline

Interactive CLI:

```bash
python run.py
```

Programmatic API:

```python
from pipeline import process

result = process("The student reads the book.")
```

The result preserves the required Stage 5 outputs:

- `corrected_text`
- `parse_trees`
- `semantic_frames`
- `discourse_relations`
- `coref_chains`

It also exposes spelling changes, POS tags, dependency parses, pragmatic inferences, and summary counts.

## Stage 1 — Spell Checking

`spell_checking/spell_checker.py` implements Levenshtein edit distance from scratch using dynamic programming. Candidate corrections combine edit distance with a shipped frequency list and a small transparent n-gram context model. Contractions are protected and accepted corrections include metadata explaining the edit distance/frequency/context evidence.

Example:

```text
I recieved the email yesterday...
```

becomes:

```text
I received the email yesterday...
```

with one spelling correction.

## Stage 2 — Syntactic Processing

spaCy supplies the POS sequence and dependency parse. The custom parser in `syntactic_processing/cfg_parser.py` is a genuine hand-written recursive-descent parser with 15 explicit CFG rules and its own tree construction.

For example, supported sentences include:

```text
The student reads the book.
The teacher checks the assignment.
The student submitted his assignment to the teacher.
The researcher reads the new report.
```

The output is a nested `S -> NP VP` tree rather than a library-generated CFG result.

## Stage 3 — Semantic Analysis

spaCy provides named entities and dependency-based semantic roles (`agent`, `action`, `patient`). WSD uses a simple Lesk-style overlap algorithm: candidate WordNet synsets are collected, definitions/lemmas/examples form each sense signature, context words are compared with those signatures, and the highest-overlap sense is selected. A small transparent domain-cue layer (for example, `deposit/cashier` for financial bank and `baseball/hit` for the sports bat) helps resolve cases where raw gloss overlap is tied or misleading. The public sense labels are kept stable across WordNet variants.

Dedicated examples cover:

- **bank** — financial institution in a deposit/cashier context
- **bat** — sports equipment in a baseball context
- **light** — illumination versus the adjective meaning “not heavy”

The result includes `word_senses` plus `wsd_details` containing the selected sense, definition, score, and overlapping context words.

## Stage 4 — Discourse & Pragmatic Processing

Coreference is a lightweight heuristic resolver using number, human/non-human compatibility, gender hints, grammatical role, subject salience, possessive preference, recency, sentence distance, and confidence margins. Resolutions are grouped into actual chains instead of counting each pronoun as a separate chain.

For example:

```text
The student submitted his assignment to the teacher.
He was worried about the result.
The teacher reviewed it and returned it to him.
```

produces chains equivalent to:

```text
student     -> his, He, him
assignment  -> it, it
```

and therefore reports **2 resolved coreference chains**.

Discourse connectives such as `however`, `but`, `because`, `therefore`, `so`, and `and` are mapped to small rule-based relation labels.

Pragmatic inference recognizes common indirect-request patterns such as:

```text
Could you send me the assignment file?
```

and returns an `indirect_request` with inferred intent `request`, action `send`, and object `assignment file`.

## Stage 5 — Full Pipeline Assembly

`pipeline.process(raw_text)` runs stages 1→4 and returns the complete structured result. `evaluate_samples.py` processes all 10 files under `tests/samples/` and prints the required summary fields.

Run:

```bash
python evaluate_samples.py
```

## Validation

The packaged project has been syntax-checked and the full automated test suite passes. The 10 supplied sample files also run successfully through the complete pipeline.

## Tests

Run the full test suite with:

```bash
python -m pytest
```

The tests cover:

- from-scratch Levenshtein distance
- spell-check regression and contraction preservation
- custom CFG parsing
- grouped coreference resolution
- bank/bat/light WSD
- indirect-request pragmatic inference
- the ten sample files
- preservation of the autocomplete bonus app

## Optional +10 — Autocomplete-as-you-type

The Streamlit bonus remains in:

```text
spell_checking/autocomplete_app.py
```

Run it with:

```bash
streamlit run spell_checking/autocomplete_app.py
```

It provides word suggestions while typing and can also display the full NLP pipeline result and CFG trees.

---

# 👨‍💻 Author

**Devansh Mankad**

Computer Engineering Student

* GitHub: https://github.com/Devansh-Mankad
---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.

---

# 📄 License

This project is licensed under the MIT License.