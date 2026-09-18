import re
from collections import OrderedDict

CONNECTIVES = {
    "however": "contrast",
    "but": "contrast",
    "because": "cause",
    "therefore": "result",
    "so": "result",
    "and": "elaboration",
}

HUMAN_NOUNS = {
    "student", "students", "teacher", "teachers", "professor", "professors",
    "manager", "managers", "person", "people", "researcher", "researchers",
    "user", "users", "player", "players", "boy", "boys", "girl", "girls",
    "man", "men", "woman", "women", "child", "children", "customer", "customers",
}
OBJECT_NOUNS = {
    "file", "files", "document", "documents", "assignment", "assignments",
    "email", "emails", "answer", "answers", "book", "books", "project", "projects",
    "report", "reports", "task", "tasks", "information", "source", "sources",
    "result", "results", "money", "cashier", "bag", "bags", "ball", "balls",
    "bat", "light",
}
ABSTRACT_NOUNS = {"result", "results", "information", "idea", "ideas", "work", "meaning", "reason", "reasons"}
MALE_NOUNS = {"boy", "man", "father", "son", "student", "player", "researcher"}
FEMALE_NOUNS = {"girl", "woman", "mother", "daughter"}
MALE_PRONOUNS = {"he", "him", "his", "himself"}
FEMALE_PRONOUNS = {"she", "her", "hers", "herself"}
PLURAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}

class DiscourseProcessor:
    def analyze(self, text, syntax):
        relations = []
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        for sent in sentences:
            for word, rel in CONNECTIVES.items():
                if re.search(r"\b" + re.escape(word) + r"\b", sent, re.I):
                    relations.append({"connective": word, "relation": rel, "sentence": sent})

        chains = self._coref(text, syntax)
        prag = []
        for sent in sentences:
            if re.search(r"^\s*(could|would|can)\s+you\b", sent, re.I) or \
               re.search(r"^\s*would\s+you\s+mind\b", sent, re.I):
                match = re.search(
                    r"^\s*(?:could|would|can)\s+you\s+(?:please\s+)?(.+?)(?:\?|$)",
                    sent, re.I
                )
                action = None
                obj = None
                if match:
                    phrase = match.group(1).strip()
                    # Lightweight extraction for common "send the file" forms.
                    am = re.match(r"(\w+)\s+(?:me\s+)?(?:the\s+)?(.+)$", phrase, re.I)
                    if am:
                        action, obj = am.group(1).lower(), am.group(2).strip()
                prag.append({
                    "sentence": sent,
                    "type": "indirect_request",
                    "inferred_intent": "request",
                    "action": action,
                    "object": obj,
                    "inference": "indirect request",
                    "explanation": "The interrogative form functions pragmatically as a request for the listener to perform the action."
                })
        return {"relations": relations, "coref_chains": chains, "pragmatic_inferences": prag}

    @staticmethod
    def _gender(noun):
        n = noun.lower()
        if n in MALE_NOUNS:
            return "male"
        if n in FEMALE_NOUNS:
            return "female"
        return "unknown"

    @staticmethod
    def _human(noun):
        n = noun.lower()
        return n in HUMAN_NOUNS

    @staticmethod
    def _number(noun):
        n = noun.lower()
        if n in {"people", "men", "women", "children"} or n.endswith("s"):
            return "plural"
        return "singular"

    def _extract_candidates(self, sentence, s_idx, syntax_item):
        """Extract noun mentions plus grammatical-role/salience features."""
        candidates = []
        pos = syntax_item.get("pos", []) if syntax_item else []
        deps = syntax_item.get("dependencies", []) if syntax_item else []
        dep_by_word = {}
        for d in deps:
            dep_by_word.setdefault(d["text"].lower(), []).append(d)

        for idx, (word, tag) in enumerate(pos):
            low = word.lower()
            if tag not in {"NOUN", "PROPN"}:
                continue
            ds = dep_by_word.get(low, [])
            dep = ds[0]["dep"] if ds else "dep"
            role = "subject" if dep in {"nsubj", "nsubjpass"} else \
                   "object" if dep in {"obj", "dobj", "pobj", "iobj", "attr"} else \
                   "prep_object" if dep in {"pobj"} else "other"
            candidates.append({
                "text": word, "lower": low, "sentence": s_idx, "token_index": idx,
                "number": self._number(word), "human": self._human(word),
                "gender": self._gender(word), "role": role
            })

        # If spaCy is unavailable, infer roles from surface position around
        # the first verb. This is deliberately conservative.
        if not candidates:
            raw = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", sentence)
            verb_positions = [i for i, w in enumerate(raw) if w.lower() in {
                "is", "was", "were", "reads", "read", "checks", "check",
                "submitted", "reviewed", "returned", "sent", "used", "picked",
                "walked", "went", "deposited", "asked"
            } or w.lower().endswith(("ed", "ing"))]
            vi = verb_positions[0] if verb_positions else len(raw)
            for i, w in enumerate(raw):
                low = w.lower()
                if low in HUMAN_NOUNS or low in OBJECT_NOUNS:
                    role = "subject" if i < vi else "object"
                    candidates.append({
                        "text": w, "lower": low, "sentence": s_idx, "token_index": i,
                        "number": self._number(w), "human": self._human(w),
                        "gender": self._gender(w), "role": role
                    })
        return candidates

    def _pronoun_requirements(self, pronoun):
        p = pronoun.lower()
        if p in MALE_PRONOUNS:
            return {"human": True, "gender": "male", "number": "singular", "possessive": p == "his"}
        if p in FEMALE_PRONOUNS:
            return {"human": True, "gender": "female", "number": "singular", "possessive": p == "her" or p == "hers"}
        if p in PLURAL_PRONOUNS:
            return {"human": None, "gender": "unknown", "number": "plural", "possessive": p == "their" or p == "theirs"}
        if p in {"it", "its"}:
            return {"human": False, "gender": "unknown", "number": "singular", "possessive": p == "its"}
        return None

    def _score(self, pronoun, ant, current_subject, sentence_index):
        req = self._pronoun_requirements(pronoun)
        if not req:
            return None, []

        # Hard compatibility constraints.
        if req["number"] != ant["number"]:
            return None, []
        if req["human"] is True and not ant["human"]:
            return None, []
        if req["human"] is False and ant["human"]:
            return None, []

        score = 0.0
        rules = []

        if req["gender"] != "unknown":
            if ant["gender"] == req["gender"]:
                score += 5
                rules.append("gender compatibility")
            elif ant["gender"] != "unknown":
                return None, []
        if req["human"] is True:
            score += 3
            rules.append("human compatibility")
        elif req["human"] is False:
            score += 2
            rules.append("non-human compatibility")

        # Grammatical role is especially important for subject pronouns and
        # possessives. A prior subject is a strong discourse antecedent.
        if ant["role"] == "subject":
            score += 5
            rules.append("subject salience")
        elif ant["role"] == "object":
            score += 2
            rules.append("object preference")
        elif ant["role"] == "prep_object":
            score += 0
            rules.append("lower prepositional-object salience")

        # For neutral "it", concrete discourse objects are preferred over
        # abstract nouns such as "result". This helps distinguish an earlier
        # document/assignment from a later abstract noun.
        if not req["human"] and ant["lower"] in ABSTRACT_NOUNS:
            score -= 1.5
            rules.append("abstract-noun penalty")
        elif not req["human"] and ant["lower"] in OBJECT_NOUNS:
            score += 1.5
            rules.append("concrete-object preference")

        if req["possessive"]:
            if ant["role"] == "subject":
                score += 2
                rules.append("possessive subject preference")

        # For object "him/her" in a coordinated construction such as
        # "teacher reviewed it and returned it to him", penalize the current
        # subject. This avoids selecting the current agent as its own recipient.
        if pronoun.lower() in {"him", "her", "them"} and current_subject and \
                ant["lower"] == current_subject["lower"]:
            score -= 5
            rules.append("current-subject penalty")

        # Recency matters, but only after grammatical/semantic compatibility.
        distance = max(0, sentence_index - ant["sentence"])
        score += max(0.0, 3.0 - 0.8 * distance)
        rules.append("recency")
        return score, rules

    def _coref(self, text, syntax):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        all_candidates = []
        for i, sentence in enumerate(sentences):
            syntax_item = syntax[i] if i < len(syntax) else {}
            all_candidates.append(self._extract_candidates(sentence, i, syntax_item))

        # Chains are keyed by antecedent mention. Multiple pronouns pointing
        # to the same antecedent become one chain, as required by the project.
        chains = OrderedDict()

        for s_idx, sentence in enumerate(sentences):
            current = all_candidates[s_idx]
            current_subject = next((c for c in current if c["role"] == "subject"), None)
            pron_matches = list(re.finditer(
                r"\b(he|she|him|her|his|hers|it|its|they|them|their|theirs)\b",
                sentence, re.I
            ))
            for pm in pron_matches:
                pron = pm.group(1)
                req = self._pronoun_requirements(pron)
                if not req:
                    continue

                # Antecedents must precede the pronoun. Prefer prior
                # sentences; same-sentence candidates are allowed only when
                # they occur before the pronoun.
                candidates = []
                for sentence_candidates in all_candidates:
                    for cand in sentence_candidates:
                        if cand["sentence"] > s_idx:
                            continue
                        if cand["sentence"] == s_idx:
                            # Find token index approximately from the word sequence.
                            prefix_words = re.findall(r"[A-Za-z]+(?:\'[A-Za-z]+)?", sentence[:pm.start()])
                            if cand["token_index"] >= len(prefix_words):
                                continue
                        candidates.append(cand)

                scored = []
                for ant in candidates:
                    score, rules = self._score(pron, ant, current_subject, s_idx)
                    if score is not None:
                        scored.append((score, ant, rules))

                if not scored:
                    continue

                scored.sort(key=lambda x: (x[0], x[1]["sentence"], x[1]["token_index"]), reverse=True)
                best = scored[0]
                # Avoid low-confidence inventions when two candidates are
                # nearly tied.
                if len(scored) > 1 and best[0] - scored[1][0] < 0.75:
                    continue

                ant, rules = best[1], best[2]
                key = (ant["sentence"], ant["token_index"], ant["lower"])
                if key not in chains:
                    chains[key] = {
                        "antecedent": ant["text"],
                        "mentions": [],
                        "resolutions": []
                    }
                chains[key]["mentions"].append(pron)
                chains[key]["resolutions"].append({
                    "pronoun": pron,
                    "antecedent": ant["text"],
                    "sentence_index": s_idx + 1,
                    "rule": " + ".join(rules)
                })

        return list(chains.values())
