"""
Module 5 — Guardrails.

Validation code that runs BEFORE every LLM call.
"""

def check_question(question: str) -> tuple[bool, str]:
    """
    Validate a user's question before sending it to the LLM.
    """

    if not question or not question.strip():
        return False, "Please enter a question."

    question = question.strip()

    # Reject questions that are too short
    if len(question) < 5:
        return False, "Please enter a more detailed question."

    # Reject questions that are too long
    if len(question) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."

    blocked_terms = [
        "hack",
        "malware",
        "virus",
        "password",
        "credit card",
        "steal",
        "illegal"
    ]

    question_lower = question.lower()

    for term in blocked_terms:
        if term in question_lower:
            return False, "I can't help with that request."

    # Block common prompt-injection phrases
    injection_phrases = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore the previous prompt",
        "forget previous instructions",
        "forget all previous instructions",
        "system prompt",
        "reveal your instructions",
        "show your instructions",
        "bypass your instructions",
        "jailbreak"
    ]

    for phrase in injection_phrases:
        if phrase in question_lower:
            return False, "I can't process that request."

    return True, ""


def check_answer(answer: str) -> str:
    """
    Ensure that an answer is not empty.
    """

    if not answer or not str(answer).strip():
        return "I don't know based on the provided career notes."

    return str(answer).strip()