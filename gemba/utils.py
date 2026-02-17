import pandas as pd
import diskcache as dc
from gemba.gpt_api import GptApi
from gemba.gemba_mqm_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer
from gemba.gemba_esa import TEMPLATE_GEMBA_ESA_ERROR_SPANS, TEMPLATE_GEMBA_ESA_RANKING
from gemba.prompt import prompts, validate_number

# Structured output schemas for OpenAI's response_format parameter.
# These force models to return valid JSON matching the schema, avoiding
# verbose free-text responses that break parsing with newer models.
_ERROR_ITEM_SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string"}, "description": {"type": "string"}},
    "required": ["category", "description"],
    "additionalProperties": False,
}

RESPONSE_FORMATS = {
    "score": {
        "type": "json_schema",
        "json_schema": {
            "name": "score_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer"}},
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    },
    "mqm": {
        "type": "json_schema",
        "json_schema": {
            "name": "mqm_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "errors": {
                        "type": "object",
                        "properties": {
                            "critical": {"type": "array", "items": _ERROR_ITEM_SCHEMA},
                            "major": {"type": "array", "items": _ERROR_ITEM_SCHEMA},
                            "minor": {"type": "array", "items": _ERROR_ITEM_SCHEMA},
                        },
                        "required": ["critical", "major", "minor"],
                        "additionalProperties": False,
                    },
                },
                "required": ["errors"],
                "additionalProperties": False,
            },
        },
    },
}


def _get_response_format(method, use_structured_output):
    """Determine the response_format for a given GEMBA method."""
    if not use_structured_output:
        return None
    if method.startswith(("GEMBA-DA", "GEMBA-SQM")):
        return RESPONSE_FORMATS["score"]
    elif method == "GEMBA-MQM":
        return RESPONSE_FORMATS["mqm"]
    return None


def get_gemba_scores(source, hypothesis, source_lang, target_lang, method, model,
                     list_mqm_errors=False, api_version=None, use_structured_output=True,
                     reference=None):
    df = pd.DataFrame({'source_seg': source, 'target_seg': hypothesis})
    df['source_lang'] = source_lang
    df['target_lang'] = target_lang
    if reference is not None:
        df['reference_seg'] = reference

    cache = dc.Cache(f'cache/{model}_{method}', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')
    gptapi = GptApi(api_version=api_version)

    response_format = _get_response_format(method, use_structured_output)

    if method == "GEMBA-MQM":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(x, list_mqm_errors=list_mqm_errors, full_desc=True)
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500, response_format=response_format)
    elif method in ["GEMBA-DA", "GEMBA-DA_ref", "GEMBA-SQM", "GEMBA-SQM_ref", "GEMBA-stars", "GEMBA-stars_ref", "GEMBA-classes", "GEMBA-classes_ref"]:
        df["prompt"] = df.apply(lambda x: apply_template(prompts[method]['prompt'], x), axis=1)
        parse_answer = prompts[method]["validate_answer"]
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache, max_tokens=500, response_format=response_format)
    elif method == "GEMBA-ESA":
        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_ERROR_SPANS, x), axis=1)
        parse_answer = lambda x: x
        error_spans = gptapi.bulk_request(df, model, parse_answer, cache=cache)
        df['error_spans'] = pd.DataFrame(error_spans)['answer']

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_ESA_RANKING, x), axis=1)
        parse_answer = validate_number
        answers = gptapi.bulk_request(df, model, parse_answer, cache=cache)
    else:
        raise Exception(f"Method {method} not supported.")

    return list(pd.DataFrame(answers)['answer'])
