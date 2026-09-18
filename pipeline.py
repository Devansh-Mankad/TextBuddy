from spell_checking.spell_checker import SpellChecker
from syntactic_processing.parser import SyntacticProcessor
from semantic_analysis.semantic import SemanticAnalyzer
from discourse_pragmatics.discourse import DiscourseProcessor

spell_checker=SpellChecker()
syntactic=SyntacticProcessor()
semantic=SemanticAnalyzer()
discourse=DiscourseProcessor()

def process(raw_text:str)->dict:
    corrected_text,changes=spell_checker.correct_text(raw_text)
    syntax=syntactic.process(corrected_text)
    semantic_frames=semantic.analyze(corrected_text,syntax)
    discourse_data=discourse.analyze(corrected_text,syntax)
    entities=sum(len(frame.get("entities",[])) for frame in semantic_frames)

    return {
        "corrected_text":corrected_text,
        "spelling_changes":changes,
        "parse_trees":[x["cfg_tree"] for x in syntax],
        "pos_tags":[x["pos"] for x in syntax],
        "dependency_parses":[x["dependencies"] for x in syntax],
        "semantic_frames":semantic_frames,
        "discourse_relations":discourse_data["relations"],
        "coref_chains":discourse_data["coref_chains"],
        "pragmatic_inferences":discourse_data["pragmatic_inferences"],
        "summary":{
            "spelling_corrections":len(changes),
            "entities_found":entities,
            "coref_chains_resolved":len(discourse_data["coref_chains"]),
            "discourse_relations_tagged":len(discourse_data["relations"])
        }
    }
