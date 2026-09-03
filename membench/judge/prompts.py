"""Judge prompts.

The LONGMEMEVAL templates are copied verbatim from get_anscheck_prompt in
src/evaluation/evaluate_qa.py of xiaowu0162/LongMemEval.

RULES["zep"] and RULES["mem0"] are copied verbatim from verify.py of
ThinkfleetAI/memmesh-benchmarks, which reproduces Zep's rule from
graphiti_core/prompts/eval.py and Mem0's partial-credit _JUDGE_TEMPLATE.
"""

from __future__ import annotations

_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model\'s response is still correct. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user\'s personal "
    "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given "
    "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
    "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)

_BY_TYPE = {
    "single-session-user": _DEFAULT,
    "single-session-assistant": _DEFAULT,
    "multi-session": _DEFAULT,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
}

RULES = {
    "strict": (
        "You grade whether a predicted answer matches the gold answer. CORRECT only "
        "if it conveys the SAME specific factual answer as the gold (exact/semantic "
        "match). Resolve numbers and dates by value (3.10%==3.1%; a relative date "
        "equals the absolute date it denotes). A vague, partial, or merely on-topic "
        'answer is INCORRECT. Reply JSON {"correct": true|false}.'
    ),
    "zep": (
        "You grade a predicted answer against a gold answer. Although the prediction "
        "may be more verbose, mark it CORRECT as long as it references the SAME TOPIC "
        'as the gold answer. Reply JSON {"correct": true|false}.'
    ),
    "mem0": (
        "You grade a predicted answer against a gold answer with PARTIAL CREDIT. Mark "
        "CORRECT if the prediction is about the same referent and conveys the gold, OR "
        "for list or how-many golds, contains AT LEAST ONE correct item from the "
        "gold list. Semantic overlap counts; extra detail is fine; resolve dates and "
        "numbers by value. INCORRECT only if wrong referent, contradictory, or a "
        'refusal. Reply JSON {"correct": true|false}.'
    ),
}

STALE_PROMPT = (
    "A fact about the user changed over time. I will give you the question, the current "
    "correct answer, and a model response that was judged incorrect. Answer yes if the "
    "response states a superseded, out-of-date value instead of the current one. Answer no "
    "if it is wrong for any other reason, such as being unrelated, a refusal, or invented.\n\n"
    "Question: {}\n\nCurrent correct answer: {}\n\nModel Response: {}\n\n"
    "Does the response give a superseded value? Answer yes or no only."
)

RULE_USER_TEMPLATE = "Question: {}\n\nGold answer: {}\n\nPredicted answer: {}"


def longmemeval_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    try:
        template = _BY_TYPE[question_type]
    except KeyError:
        raise ValueError(f"Unknown question_type: {question_type}")
    return template.format(question, answer, response)
