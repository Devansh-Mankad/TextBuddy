import os
import unittest

from pipeline import process
from spell_checking.spell_checker import SpellChecker
from syntactic_processing.cfg_parser import CFGParser
from semantic_analysis.semantic import SemanticAnalyzer
from discourse_pragmatics.discourse import DiscourseProcessor


class TestAssistant(unittest.TestCase):
    def setUp(self):
        self.checker = SpellChecker()

    def test_levenshtein(self):
        self.assertEqual(SpellChecker.levenshtein("kitten", "sitting"), 3)

    def test_does_not_destroy_contraction(self):
        corrected, _ = self.checker.correct_text("I didn't understand what they wanted me to do.")
        self.assertIn("didn't", corrected)
        self.assertIn("what", corrected)
        self.assertIn("me", corrected)

    def test_common_typo(self):
        corrected, changes = self.checker.correct_text("I recieved the email.")
        self.assertIn("received", corrected)
        self.assertEqual(changes[0]["original"], "recieved")

    def test_required_spell_check_regression(self):
        text = ("I recieved the email yesterday, but I didn't understand what they wanted "
                "me to do. However, I will check the document again and send a reply tomorrow.")
        corrected, changes = self.checker.correct_text(text)
        expected = ("I received the email yesterday, but I didn't understand what they wanted "
                    "me to do. However, I will check the document again and send a reply tomorrow.")
        self.assertEqual(corrected, expected)
        self.assertEqual(len(changes), 1)

    def test_does_not_change_known_words(self):
        corrected, changes = self.checker.correct_text("Hello Good Morning. What will you help me with?")
        self.assertEqual(corrected, "Hello Good Morning. What will you help me with?")
        self.assertEqual(changes, [])

    def test_pipeline_keys(self):
        result = process("The student recieve the file. He sent it because the teacher asked.")
        for key in ["corrected_text", "spelling_changes", "parse_trees", "pos_tags",
                    "dependency_parses", "semantic_frames", "discourse_relations",
                    "coref_chains", "pragmatic_inferences", "summary"]:
            self.assertIn(key, result)

    def test_coreference_grouped_chains(self):
        text = ("The student submitted his assignment to the teacher. "
                "He was worried about the result. "
                "The teacher reviewed it and returned it to him.")
        result = process(text)
        chains = result["coref_chains"]
        by_ant = {c["antecedent"].lower(): c for c in chains}
        self.assertIn("student", by_ant)
        self.assertIn("assignment", by_ant)
        self.assertEqual(by_ant["student"]["mentions"], ["his", "He", "him"])
        self.assertEqual(by_ant["assignment"]["mentions"], ["it", "it"])
        self.assertEqual(result["summary"]["coref_chains_resolved"], 2)
        for chain in chains:
            for resolution in chain["resolutions"]:
                self.assertIn("rule", resolution)

    def test_cfg_simple_declaratives(self):
        examples = [
            "The student reads the book.",
            "The teacher checks the assignment.",
            "The student submitted his assignment to the teacher.",
            "The teacher reviewed the document.",
            "The researcher reads the new report.",
        ]
        for sentence in examples:
            result = process(sentence)
            tree = result["parse_trees"][0]
            self.assertEqual(tree["label"], "S")
            self.assertFalse(any(c.get("label") == "UNPARSED" for c in tree["children"]))

    def test_cfg_multisentence_regression(self):
        text = ("The student submitted his assignment to the teacher. "
                "He was worried about the result. "
                "The teacher reviewed it and returned it to him.")
        trees = process(text)["parse_trees"]
        self.assertEqual(len(trees), 3)
        for tree in trees:
            self.assertEqual(tree["label"], "S")
            self.assertFalse(any(c.get("label") == "UNPARSED" for c in tree["children"]))

    def test_cfg_parser_is_hand_built(self):
        tree = CFGParser([
            ("The", "DET"), ("student", "NOUN"), ("reads", "VERB"),
            ("the", "DET"), ("book", "NOUN")
        ]).parse()
        self.assertEqual(tree["label"], "S")
        self.assertEqual(tree["children"][0]["label"], "NP")
        self.assertEqual(tree["children"][1]["label"], "VP")

    def test_wsd_bank(self):
        analyzer = SemanticAnalyzer()
        context = ("I went to the bank to deposit my money. The bank was crowded "
                   "because many customers were waiting for the cashier.")
        sense = analyzer.wsd_lesk("bank", context)
        self.assertIn(sense, {"bank.n.02", "bank.n.01"})
        # In both real WordNet and fallback mode, the selected financial sense
        # is distinguishable by its definition/details.
        details = analyzer.wsd_details("bank", context)
        self.assertTrue("financial" in details["definition"].lower() or "deposit" in details["definition"].lower())

    def test_wsd_bat(self):
        analyzer = SemanticAnalyzer()
        context = ("The boy picked up the bat and walked toward the baseball field. "
                   "He used the bat to hit the ball.")
        details = analyzer.wsd_details("bat", context)
        self.assertIn(details["sense"], {"bat.n.02"})
        self.assertTrue("club" in details["definition"].lower() or "hit" in details["definition"].lower())

    def test_wsd_light(self):
        analyzer = SemanticAnalyzer()
        context = ("The room was filled with bright light from the window. "
                   "The light bag was easy to carry.")
        details = analyzer.wsd_details("light", context)
        self.assertIn(details["sense"], {"light.n.01", "light.a.01"})
        self.assertTrue(details["definition"])

    def test_pragmatic_indirect_request(self):
        result = process("Could you send me the assignment file?")
        inferences = result["pragmatic_inferences"]
        self.assertTrue(inferences)
        inf = inferences[0]
        self.assertEqual(inf["type"], "indirect_request")
        self.assertEqual(inf["inferred_intent"], "request")
        self.assertEqual(inf["action"], "send")
        self.assertEqual(inf["object"], "assignment file")

    def test_ten_samples_exist(self):
        for i in range(1, 11):
            self.assertTrue(os.path.exists(f"tests/samples/sample_{i}.txt"))

    def test_bonus_app_exists(self):
        self.assertTrue(os.path.exists("spell_checking/autocomplete_app.py"))


if __name__ == "__main__":
    unittest.main()
