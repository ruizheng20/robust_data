# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : build_attacker_utils.py
@Project  : Robust_Data
@Time     : 2022/10/1 12:35 下午
@Author   : Zhiheng Xi
"""
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import InputColumnModification
from textattack.goal_functions import UntargetedClassification
from textattack.attack_recipes import (PWWSRen2019,
                                       BAEGarg2019,
                                       TextBuggerLi2018,
                                       TextFoolerJin2019,
                                       PSOZang2020,
                                       CLARE2020,
                                       )
from textattack import Attack

def build_english_attacker( model,attack_method) -> Attack:
    attacker = None
    if attack_method== 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif attack_method== 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif attack_method== 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif attack_method== 'pwws':
        attacker = PWWSRen2019.build(model)
    elif attack_method == 'pso':
        attacker = PSOZang2020.build(model)
    elif attack_method== 'clare':
        attacker = CLARE2020.build(model)

    else:
        print("Not implement attck!")
        exit(41)


    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def build_weak_attacker(neighbour_vocab_size,sentence_similarity, model,attack_method) -> Attack:
    attacker = None
    if attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif attack_method == 'textfooler' :
        attacker = TextFoolerJin2019.build(model)
    elif attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print("Not implement attck!")
        exit(41)


    if attack_method in ['bertattack','textfooler']:
        attacker.transformation = WordSwapEmbedding(max_candidates=neighbour_vocab_size)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
            if isinstance(constraint, UniversalSentenceEncoder):
                attacker.constraints.remove(constraint)

    # attacker.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=sentence_similarity,
        metric="cosine",
        compare_against_original=True,
        window_size=15,
        skip_text_shorter_than_window=False,
    )
    attacker.constraints.append(use_constraint)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    attacker.goal_function = UntargetedClassification(model)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)

