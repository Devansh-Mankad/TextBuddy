"""Stage 2b — hand-built recursive-descent CFG parser.

This parser deliberately uses a small explicit grammar and builds the tree
itself.  It is not a wrapper around spaCy's dependency parser.  The grammar
covers the simple declaratives used by the project, including auxiliary +
adjective predicates and coordinated verb phrases.
"""


class CFGParser:
    # 15 explicit CFG productions used by the parser.
    RULES = [
        "S -> NP VP",
        "S -> NP VP CONJ VP",
        "NP -> PRON",
        "NP -> PRON NBAR",
        "NP -> DET NBAR",
        "NP -> NBAR",
        "NBAR -> ADJ NBAR",
        "NBAR -> N",
        "NBAR -> N NBAR",
        "VP -> V",
        "VP -> V NP",
        "VP -> V NP PP",
        "VP -> V NP CONJ VP",
        "VP -> AUX ADJ PP",
        "PP -> P NP",
    ]

    def __init__(self, tokens):
        self.t = [(text, tag) for text, tag in tokens if tag != "PUNCT"]
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def consume(self, tags):
        if self.i >= len(self.t):
            return None
        allowed = {tags} if isinstance(tags, str) else set(tags)
        if self.t[self.i][1] in allowed:
            item = self.t[self.i]
            self.i += 1
            return item
        return None

    @staticmethod
    def node(label, children):
        return {"label": label, "children": children}

    @staticmethod
    def leaf(tag, word):
        return {"label": tag, "children": [word]}

    def _attempt(self, fn):
        """Run a production with backtracking on failure."""
        start = self.i
        result = fn()
        if result is None:
            self.i = start
        return result

    def np(self):
        # NP -> PRON NBAR (possessive) | PRON
        def possessive():
            pron = self.consume("PRON")
            if not pron:
                return None
            if self.peek() and self.peek()[1] in {"ADJ", "NOUN", "PROPN"}:
                nb = self.nbar()
                if nb:
                    return self.node("NP", [self.leaf("PRON", pron[0]), nb])
            return None

        result = self._attempt(possessive)
        if result:
            return result

        pron = self.consume("PRON")
        if pron:
            return self.node("NP", [self.leaf("PRON", pron[0])])

        # NP -> DET NBAR
        def determiner_np():
            det = self.consume("DET")
            if not det:
                return None
            nb = self.nbar()
            if nb:
                return self.node("NP", [self.leaf("DET", det[0]), nb])
            return None

        result = self._attempt(determiner_np)
        if result:
            return result

        # NP -> NBAR
        nb = self.nbar()
        if nb:
            return self.node("NP", [nb])
        return None

    def nbar(self):
        start = self.i
        children = []

        # NBAR -> ADJ NBAR (allow multiple adjectives)
        while self.peek() and self.peek()[1] == "ADJ":
            adj = self.consume("ADJ")
            children.append(self.leaf("ADJ", adj[0]))

        noun = self.consume({"NOUN", "PROPN"})
        if not noun:
            self.i = start
            return None
        children.append(self.leaf("N", noun[0]))

        # NBAR -> N NBAR, e.g. "assignment file" / "research report".
        if self.peek() and self.peek()[1] in {"NOUN", "PROPN", "ADJ"}:
            rhs = self._attempt(self.nbar)
            if rhs:
                children.append(rhs)

        return self.node("NBAR", children)

    def pp(self):
        start = self.i
        prep = self.consume("ADP")
        if not prep:
            return None
        np = self.np()
        if np:
            return self.node("PP", [self.leaf("P", prep[0]), np])
        self.i = start
        return None

    def vp(self):
        start = self.i

        # VP -> AUX ADJ PP, e.g. "was worried about the result".
        aux = self.consume("AUX")
        if aux:
            adj = self.consume("ADJ")
            if adj:
                children = [self.leaf("AUX", aux[0]), self.leaf("ADJ", adj[0])]
                pp = self.pp()
                if pp:
                    children.append(pp)
                # Only accept this production if it consumes a useful
                # predicate; an unmatched trailing token will be rejected by
                # parse(), which then returns an explicit UNPARSED tree.
                return self.node("VP", children)
            self.i = start

        # VP -> V ...
        verb = self.consume("VERB")
        if not verb:
            return None
        children = [self.leaf("V", verb[0])]

        # VP -> V NP ...
        np = self.np()
        if np:
            children.append(np)

            # VP -> V NP CONJ VP, e.g. "reviewed it and returned it to him".
            conj = self._attempt(lambda: self.consume("CCONJ"))
            if conj:
                rhs = self.vp()
                if rhs:
                    children.extend([self.leaf("CONJ", conj[0]), rhs])
                    return self.node("VP", children)
                # No RHS: backtrack and leave conjunction for parse failure.
                self.i -= 1

            # Optional PP after the object: "submitted the assignment to the teacher".
            pp = self.pp()
            if pp:
                children.append(pp)
        else:
            # VP -> V
            pass

        return self.node("VP", children)

    def parse(self):
        self.i = 0
        if not self.t:
            return self.node("S", [])

        subject = self.np()
        predicate = self.vp()

        if subject and predicate:
            # Top-level coordination is also accepted if the VP parser did
            # not consume it internally.
            if self.i == len(self.t):
                return self.node("S", [subject, predicate])

            start = self.i
            conj = self.consume("CCONJ")
            if conj:
                rhs = self.vp()
                if rhs and self.i == len(self.t):
                    return self.node("S", [subject, predicate,
                                            self.leaf("CONJ", conj[0]), rhs])
            self.i = start

        return self.node("S", [
            self.node("UNPARSED", [word for word, _ in self.t])
        ])
