import logging
import os
import sys
import time

import openai
import tqdm
from openai import BadRequestError, NotFoundError, PermissionDeniedError

logger = logging.getLogger(__name__)


# class for calling OpenAI API and handling cache
class GptApi:
    def __init__(self, verbose=False, api_version=None):
        self.verbose = verbose

        if "OPENAI_AZURE_ENDPOINT" in os.environ:
            assert "OPENAI_AZURE_KEY" in os.environ, "OPENAI_AZURE_KEY not found in environment"

            # Azure API access
            self.client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_AZURE_KEY"],
                azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
                api_version=api_version or "2023-07-01-preview",
            )
        elif "OPENAI_API_KEY" in os.environ:
            # OpenAI API access
            self.client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"]
            )
        else:
            raise Exception("OPENAI_API_KEY or OPENAI_AZURE_KEY not found in environment")

        # Suppress noisy HTTP loggers (don't touch the root logger)
        for _name in ("httpx", "openai", "urllib3"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    # answer_id is used for determining if it was the top answer or how deep in the list it was
    def request(self, prompt, model, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None, response_format=None):
        request = {"model": model, "temperature": temperature, "prompt": prompt}

        if request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens, response_format=response_format)
            cache[request] = answers

        # there is no valid answer
        if len(answers) == 0:
            return [{
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": None,
                    "model": model,
                    }]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if self.verbose:
                logger.debug("Answer (t=%d): %s (%s)", temperature, answer, full_answer)
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase temperature and try again
        if len(parsed_answers) == 0:
            return self.request(prompt, model, parse_response, temperature=temperature + 1, answer_id=answer_id, cache=cache, response_format=response_format)

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=None, response_format=None):
        if temperature > 10:
            return []

        while True:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens, response_format=response_format)
                break
            except (BadRequestError, NotFoundError, PermissionDeniedError) as e:
                if getattr(e, "code", None) == "content_filter":
                    return []
                raise
            except Exception as e:
                error_body = getattr(e, "error", None)
                if isinstance(error_body, dict) and error_body.get("code") == "invalid_model_output":
                    return []
                logger.warning("API error, retrying: %s", e)
                time.sleep(1)

        answers = []
        for choice in response.choices:
            if choice.message.content is None:
                return []
            if hasattr(choice, "message"):
                answer = choice.message.content.strip()
            else:
                answer = choice.text.strip()

            # one of the responses didn't finish, we need to request more tokens
            if choice.finish_reason != "stop":
                logger.warning("Finish reason: %s", choice.finish_reason)
                if max_tokens is None:
                    return []
                return self.request_api(prompt, model, temperature=temperature, max_tokens=max_tokens + 200, response_format=response_format)

            answers.append({
                "answer": answer,
                "finish_reason": choice.finish_reason,
            })

        if len(answers) > 1:
            # remove duplicate answers
            answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

        return answers

    def call_api(self, prompt, model, temperature, max_tokens, response_format=None):
        parameters = {
            "temperature": temperature/10,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "model": model
        }

        if response_format is not None:
            parameters["response_format"] = response_format

        if max_tokens is not None:
            if any(model.startswith(p) for p in ("gpt-4.1", "gpt-4o", "gpt-5")):
                parameters["max_completion_tokens"] = max_tokens
            else:
                parameters["max_tokens"] = max_tokens

        if isinstance(prompt, list):
            # check that prompt contain list of dictionaries with role and content
            assert all(isinstance(p, dict) for p in prompt), "Prompts must be a list of dictionaries."
            assert all("role" in p and "content" in p for p in prompt), "Prompts must be a list of dictionaries with role and content."

            parameters["messages"] = prompt
        else:
            parameters["messages"] = [{
                "role": "user",
                "content": prompt,
            }]

        return self.client.chat.completions.create(**parameters)

    def bulk_request(self, df, model, parse_mqm_answer, cache, max_tokens=None, response_format=None):
        answers = []
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stderr):
            prompt = row["prompt"]
            parsed_answers = self.request(prompt, model, parse_mqm_answer, cache=cache, max_tokens=max_tokens, response_format=response_format)
            answers += parsed_answers
        return answers
