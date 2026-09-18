import re
import spacy
from .cfg_parser import CFGParser

# Minimal fallback tagger used only when en_core_web_sm is unavailable. The
# assignment's primary POS/dependency route remains spaCy; this keeps the
# lightweight demo and custom CFG usable in a clean installation.
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "him", "her", "us", "them",
            "his", "hers", "its", "their", "theirs", "me", "who", "what"}
CONJUNCTIONS = {"and", "but", "or", "because", "however", "therefore", "so"}
PREPOSITIONS = {"to", "in", "on", "at", "from", "with", "for", "of", "by", "into", "over", "toward", "towards", "about"}
AUXILIARIES = {"am", "is", "are", "was", "were", "be", "been", "being", "will", "would", "can", "could", "should", "may", "might", "must"}
COMMON_ADJECTIVES = {"new", "bright", "light", "worried", "crowded", "easy", "useful", "clear", "complete", "good", "many"}
COMMON_VERBS = {"read", "reads", "check", "checks", "submit", "submits", "submitted", "review", "reviews",
                "reviewed", "return", "returns", "returned", "send", "sends", "sent", "pick", "picked",
                "walk", "walked", "use", "used", "hit", "deposit", "deposited", "go", "went", "carry",
                "carries", "asked", "ask", "help", "want", "wanted", "finish", "finished", "said",
                "contain", "contains", "compare", "compared", "continued", "writing", "write", "wrote"}
COMMON_ADVERBS = {"quickly", "slowly", "yesterday", "tomorrow", "again", "carefully", "easily"}

def _fallback_pos_tag(tokens):
    tags = []
    for i, raw in enumerate(tokens):
        w = raw.lower()
        if not re.match(r"[A-Za-z]+(?:'[A-Za-z]+)?$", raw):
            tag = "PUNCT"
        elif w in DETERMINERS:
            tag = "DET"
        elif w in PRONOUNS:
            tag = "PRON"
        elif w in CONJUNCTIONS:
            tag = "CCONJ"
        elif w in PREPOSITIONS:
            tag = "ADP"
        elif w in AUXILIARIES:
            tag = "AUX"
        elif w in COMMON_ADVERBS or w.endswith("ly"):
            tag = "ADV"
        elif w in COMMON_ADJECTIVES:
            tag = "ADJ"
        elif w in COMMON_VERBS or w.endswith(("ed", "ing")):
            tag = "VERB"
        elif raw[:1].isupper() and i > 0:
            tag = "PROPN"
        else:
            tag = "NOUN"
        tags.append((raw, tag))
    return tags

class SyntacticProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.available = True
        except Exception:
            self.nlp = None
            self.available = False

    @staticmethod
    def _sentences(text):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _fallback_process(self, text):
        out = []
        for sentence in self._sentences(text):
            raw_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,!?;:]", sentence)
            pos = _fallback_pos_tag(raw_tokens)
            tree = CFGParser(pos).parse()
            # A small dependency representation for demo robustness. It is
            # explicitly marked fallback; when spaCy is installed, its actual
            # dependency parser is used instead.
            word_pos = [(w, t) for w, t in pos if t != "PUNCT"]
            root_idx = next((i for i, (_, t) in enumerate(word_pos) if t in {"VERB", "AUX"}), None)
            deps = []
            for i, (w, tag) in enumerate(word_pos):
                dep = "dep"
                head = word_pos[root_idx][0] if root_idx is not None else w
                if root_idx is not None and i < root_idx and tag in {"NOUN", "PROPN", "PRON"}:
                    dep = "nsubj"
                elif root_idx is not None and i > root_idx and tag in {"NOUN", "PROPN", "PRON"}:
                    dep = "obj"
                elif root_idx is not None and tag == "ADP":
                    dep = "prep"
                elif root_idx is not None and i == root_idx:
                    dep = "ROOT"
                deps.append({"text": w, "pos": tag, "dep": dep, "head": head})
            out.append({"sentence": sentence, "pos": pos, "dependencies": deps, "cfg_tree": tree,
                        "parser_backend": "fallback-pos-for-demo"})
        return out

    def process(self, text):
        if self.nlp is None:
            return self._fallback_process(text)

        doc = self.nlp(text)
        out = []
        for sent in doc.sents:
            tokens = [(t.text, t.pos_) for t in sent]
            cfg = CFGParser(tokens).parse()
            deps = [{"text": t.text, "pos": t.pos_, "dep": t.dep_, "head": t.head.text} for t in sent]
            out.append({"sentence": sent.text, "pos": tokens, "dependencies": deps,
                        "cfg_tree": cfg, "parser_backend": "spacy-en_core_web_sm"})
        return out
