"""
Stage 1 — From-scratch edit-distance spell checker.

The Levenshtein algorithm is implemented manually with dynamic programming.
Candidate selection combines edit distance, word frequency, and a small
bigram/trigram language model. Contractions are treated as atomic tokens so
"didn't" can never become "didn ' t".
"""
import json, math, re
from collections import Counter, defaultdict
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:[.,][0-9]+)?|[^\w\s]")
CONTRACTIONS = {
    "ain't","aren't","can't","couldn't","didn't","doesn't","don't","hadn't",
    "hasn't","haven't","he'd","he'll","he's","i'd","i'll","i'm","i've",
    "isn't","it'd","it'll","it's","let's","mustn't","shan't","she'd",
    "she'll","she's","shouldn't","that's","there's","they'd","they'll",
    "they're","they've","wasn't","we'd","we'll","we're","we've","weren't",
    "what's","where's","who's","won't","wouldn't","you'd","you'll","you're",
    "you've"
}

# A compact corpus gives the project a transparent, reproducible context model.
CORPUS = """
the student submitted the assignment before the deadline
the teacher reviewed the assignment in the morning
i received the email yesterday and sent a reply
i will check the document again tomorrow
please send the file to the teacher
could you send the file to me
would you send the report to the teacher
the students finished their work before the class
the student said that he wanted to finish the assignment
she wrote the answer and they reviewed it
the teacher said that the ideas were good
there was a grammatical mistake and a spelling error
the research notes contained confusing information
the source was reliable and the information was useful
the bank approved the loan for the customer
the bank was beside the river
the bat flew over the bank
the player used a bat in the game
the light in the room was bright
the bag was light enough to carry
i left the book there on the table
i left there on the table
i put their books on the table
their books were on the table
they submitted their work before the deadline
he checked his email in the morning
she checked her email in the morning
the file was attached to the email
however the answer was incomplete
therefore the student revised the answer
the student continued writing because the deadline was near
the teacher reviewed the work but the student was not satisfied
the document was confusing but the conclusion was clear
good morning and have a nice day
hello good morning
please help me with this assignment
what do you want me to do
i did not understand what they wanted me to do
we can check the document tomorrow
the report contains several useful examples
the project team completed the task
the researcher compared two reliable sources
the student sent the document to his teacher
the teacher asked the student to correct the errors
the students discussed the project and submitted it
the manager said that the report was ready
the user asked whether the file was available
the answer was clear and complete
the paragraph contains a few spelling errors
the application should preserve the original meaning
the assistant should correct only genuine spelling errors
"""
def _build_ngrams(corpus):
    toks = re.findall(r"[a-z]+(?:'[a-z]+)?", corpus.lower())
    uni=Counter(toks); bi=Counter(zip(toks,toks[1:])); tri=Counter(zip(toks,toks[1:],toks[2:]))
    return uni,bi,tri

class SpellChecker:
    def __init__(self, vocabulary=None, frequency_path=None):
        if frequency_path is None:
            frequency_path = Path(__file__).with_name("word_frequencies.json")
        with open(frequency_path, encoding="utf-8") as f:
            self.frequencies = {k:int(v) for k,v in json.load(f).items()}
        self.vocabulary = set(vocabulary or self.frequencies)
        # Add common contractions and project-specific words.
        self.vocabulary.update(CONTRACTIONS)
        self.vocabulary.update({
            "nlp","spacy","wordnet","cfg","cky","lesk","streamlit","autocomplete",
            "coreference","pragmatic","discourse","semantic","syntactic"
        })
        self.unigram, self.bigram, self.trigram = _build_ngrams(CORPUS)
        self.by_length=defaultdict(list)
        for w in self.vocabulary:
            if w.isalpha():
                self.by_length[len(w)].append(w)

    @staticmethod
    def levenshtein(a,b):
        a,b=a.lower(),b.lower()
        prev=list(range(len(b)+1))
        for i,ca in enumerate(a,1):
            cur=[i]
            for j,cb in enumerate(b,1):
                cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(ca!=cb)))
            prev=cur
        return prev[-1]

    def candidates(self, word, max_distance=None, limit=32):
        w=word.lower()
        if max_distance is None:
            max_distance=1 if len(w)<=4 else 2
        pool=[]
        for L in range(max(1,len(w)-max_distance),len(w)+max_distance+1):
            pool.extend(self.by_length.get(L,[]))
        scored=[]
        for cand in set(pool):
            if cand==w or "'" in cand or not cand.isalpha():
                continue
            d=self.levenshtein(w,cand)
            if d<=max_distance:
                scored.append((cand,d))
        return sorted(scored,key=lambda x:(x[1],-self.frequencies.get(x[0],0),x[0]))[:limit]

    def _context_score(self,cand,left,right):
        # Smoothed log-probability-like score. It is intentionally simple and
        # inspectable for an academic demonstration.
        score=0.7*math.log1p(self.frequencies.get(cand,0))
        # Context is deliberately weighted strongly enough to resolve
        # equal-distance alternatives such as "thare" -> "there"/"their".
        if left:
            score += 5.0*math.log1p(self.bigram.get((left.lower(),cand),0))
        if right:
            score += 5.0*math.log1p(self.bigram.get((cand,right.lower()),0))
        if left and right:
            score += 3.0*math.log1p(self.trigram.get((left.lower(),cand,right.lower()),0))
        return score

    def choose_candidate(self, word, left="", right=""):
        lower=word.lower()
        cands=self.candidates(word)
        if not cands:
            return word,None,0.0
        # Require a plausible typo: very short unknown tokens are left alone
        # unless they are a high-confidence one-edit typo.
        best=None
        for cand,d in cands:
            score=self._context_score(cand,left,right) - 1.8*d
            item=(score,cand,d)
            if best is None or item>best:
                best=item
        score,cand,d=best
        # Context/frequency confidence threshold prevents "what" -> "bat",
        # "will" -> "file", etc. when the original is actually unknown only
        # because of a missing dictionary entry.
        runner_scores=[]
        for c,dd in cands:
            runner_scores.append(self._context_score(c,left,right)-1.8*dd)
        runner=sorted(runner_scores,reverse=True)[1] if len(runner_scores)>1 else score-2.0
        margin=score-runner
        if d>2 or (len(lower)<=2) or margin<0.65:
            return word,None,margin
        reason=f"edit-distance={d}, frequency={self.frequencies.get(cand,0)}, context-score={score:.2f}"
        return cand,reason,margin

    def correct_text(self,text):
        tokens=_TOKEN_RE.findall(text)
        changes=[]
        word_indices=[i for i,t in enumerate(tokens) if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?",t)]
        for idx in word_indices:
            word=tokens[idx]
            lower=word.lower()
            if lower in self.vocabulary or lower in CONTRACTIONS:
                continue
            left=next((tokens[j] for j in range(idx-1,-1,-1)
                       if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?",tokens[j])), "")
            right=next((tokens[j] for j in range(idx+1,len(tokens))
                        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?",tokens[j])), "")
            repl,reason,_=self.choose_candidate(word,left,right)
            if reason and repl.lower()!=lower:
                if word.isupper(): repl=repl.upper()
                elif word[0].isupper(): repl=repl.capitalize()
                tokens[idx]=repl
                changes.append({
                    "original":word,
                    "corrected":repl,
                    "edit_distance":self.levenshtein(word,repl),
                    "reason":reason
                })
        # Reconstruct without spaces before punctuation, while preserving
        # normal spaces around words.
        out=""
        no_space_before={".",",","!","?",";",":",")","]","}"}
        no_space_after={"(","[","{"}
        for t in tokens:
            if not out:
                out=t
            elif t in no_space_before:
                out+=t
            elif out[-1] in no_space_after:
                out+=t
            else:
                out+=" "+t
        return out,changes
