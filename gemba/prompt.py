import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_and_check_numerical_answer(answer, min=None, max=None):
    attempt = parse_numerical_answer(answer, min, max)
    if attempt is not None:
        if attempt < min or attempt > max:
            return None
        return attempt

    return None


def parse_numerical_answer(answer, min=None, max=None):
    # Try structured JSON output first (from response_format)
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, dict) and "score" in parsed:
            return int(parsed["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # --- Original parsing logic ---
    # get all numbers in a string
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 1:
        return int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    # --- Extended fallbacks for newer models that return verbose responses ---

    # Strip markdown bold formatting and retry
    cleaned = re.sub(r"\*+", "", answer).strip()
    if cleaned != answer:
        numbers = re.findall(r'\d+', cleaned)
        if len(numbers) == 1:
            return int(numbers[0])
        if max is not None:
            r2 = re.match(rf"^[0-9]*/{max}$", cleaned)
            if r2 is not None:
                return int(cleaned.split("/")[0])

    # Look for N/MAX pattern anywhere in the string (original only matches ^N/MAX$)
    if max is not None:
        fraction_match = re.search(rf"(\d+)/{max}(?:\D|$)", cleaned)
        if fraction_match is not None:
            return int(fraction_match.group(1))

    # If exactly 2 numbers and the second equals max, take the first as the score
    numbers = re.findall(r"\d+", cleaned)
    if len(numbers) == 2 and max is not None:
        vals = [int(n) for n in numbers]
        if vals[1] == max:
            return vals[0]

    return None


def validate_number(x, min=0, max=100):
    attempt = parse_and_check_numerical_answer(x, min, max)
    if attempt is not None:
        return attempt
    return None


def parse_classes(answer, classes):
    final_class = None
    for i in range(len(classes)):
        if classes[i].lower() in answer.lower():
            if final_class is None:
                final_class = i
            else:
                logger.warning("Two classes found in answer: %s", answer)
                return None

    return final_class


def validate_stars(x):
    # Strip markdown bold formatting before processing (newer models wrap in **)
    x = re.sub(r"\*\*(.+?)\*\*", r"\1", x)
    x = x.lower()
    # try to find all possible answers as sometimes it seems to be explaining itself
    possible_answers = set()

    # check if string x contains * characters
    if "*" in x:
        possible_answers.add(x.count("*"))
    if "★" in x:
        possible_answers.add(x.count("★"))

    x = f" {x} ".replace("\n", " ")
    # possible answers: "five stars", "5 stars", "five", "five starts: perfect translation", ...
    if " one " in x or "1 star" in x:
        possible_answers.add(1)
    if " two " in x or "2 star" in x:
        possible_answers.add(2)
    if " three " in x or "3 star" in x:
        possible_answers.add(3)
    if " four " in x or "4 star" in x:
        possible_answers.add(4)
    if " five " in x or "5 star" in x:
        possible_answers.add(5)

    numerical = parse_numerical_answer(x)
    if numerical is not None:
        possible_answers.add(numerical)

    if len(possible_answers) == 1:
        answer = possible_answers.pop()
        if 1 <= answer <= 5:
            return answer
    return None


language_codes = {
    "en": "English",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}

prompts = {
    "GEMBA-DA": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-DA_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: {reference_seg}\n{target_lang} machine translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True},

    "GEMBA-SQM": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-SQM_ref": {
        "prompt": 'Score the following machine translation from {source_lang} to {target_lang} with respect to the human reference on a continuous scale from 0 to 100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} machine translation: "{target_seg}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True},

    "GEMBA-stars": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": False},

    "GEMBA-stars_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to the human reference with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": True},

    "GEMBA-classes": {
        "prompt": 'Classify the quality of machine translation from {source_lang} to {target_lang} into one of following classes: "No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation".\n\n{source_lang} source: "{source_seg}"\n{target_lang} machine translation: "{target_seg}"\nClass: ',
        "use_ref": False,
        "validate_answer": lambda x, classes=["No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation"]: parse_classes(x, classes),
        "max_tokens": 100},

    "GEMBA-classes_ref": {
        "prompt": 'Classify the quality of machine translation from {source_lang} to {target_lang} with respect to the human reference into one of following classes: "No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} machine translation: "{target_seg}"\nClass: ',
        "use_ref": True,
        "validate_answer": lambda x, classes=["No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation"]: parse_classes(x, classes),
        "max_tokens": 100},
}
