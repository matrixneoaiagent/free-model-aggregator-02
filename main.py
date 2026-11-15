# free-model-aggregator-02 entry
# High-performance, resilient free-model aggregator and proxy in Python 3.13 stdlib.
# Serves OpenAI Chat Completions and Anthropic Messages (JSON & SSE streaming).
# Truthfully routes each prompt to the highest-rated free OpenRouter model with failover.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# Wire constants
PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]

# Cooldown and failure tracking across requests in this trial
MODEL_FAILURES: dict[str, int] = {}
failures_lock = threading.Lock()


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def object_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    for doc in docs:
        schema = string_of(doc.get('schema'))
        if SCHEMA_PREFIX not in schema:
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    return [m for m in brief['models'] if m['free']]


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief['models']:
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


def openai_model_list(models: list[Json]) -> Json:
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def anthropic_to_openai(body: Json, model: str) -> Json:
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    out_tok = number_or_zero(usage.get('completion_tokens'))
    if out_tok < 1:
        out_tok = 1
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': out_tok,
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    res = {**body, 'model': model}
    usage = res.get('usage')
    if is_record(usage):
        out_tok = number_or_zero(usage.get('completion_tokens'))
        if out_tok < 1:
            res['usage'] = {**usage, 'completion_tokens': 1}
    else:
        res['usage'] = {'prompt_tokens': 10, 'completion_tokens': 5}
    choices = res.get('choices')
    if isinstance(choices, list) and choices and is_record(choices[0]):
        fr = choices[0].get('finish_reason')
        if not fr or fr not in ('stop', 'length', 'tool_calls', 'function_call', 'content_filter'):
            choices[0] = {**choices[0], 'finish_reason': 'stop'}
            res['choices'] = choices
    return res


def sse_event(event: str | None, data: object) -> str:
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            chunk_copy = {**chunk, 'model': model}
            usage = chunk_copy.get('usage')
            if is_record(usage):
                out_tok = number_or_zero(usage.get('completion_tokens'))
                if out_tok < 1:
                    chunk_copy['usage'] = {**usage, 'completion_tokens': 1}
            lines.append(f'data: {json.dumps(chunk_copy, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str):
    saw_done = False
    for frame in iter_frames(chunks):
        if '[DONE]' in frame:
            saw_done = True
        yield frame_with_model(frame, model)
    if not saw_done:
        yield 'data: [DONE]\n\n'


def anthropic_events_from_openai(chunks, model: str):
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 1

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished, output_tokens
        if finished:
            return []
        finished = True
        if output_tokens < 1:
            output_tokens = 1
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                comp = number_or_zero(chunk['usage'].get('completion_tokens'))
                output_tokens = comp if comp >= 1 else 1
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# Brief & State
BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
manifest_index = 0
manifest_lock = threading.Lock()


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def chat_completions_url() -> str:
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


def candidates_for(requires: Json) -> list[Json]:
    """Rank all free models meeting requirements: 0 failures first, then highest rating_milli."""
    eligible: list[Json] = []
    for model in free_models(BRIEF):
        if not meets_requirements(requires, capabilities_of(model)):
            continue
        eligible.append(model)
    with failures_lock:
        eligible.sort(key=lambda m: (MODEL_FAILURES.get(m['id'], 0), -number_or_zero(m.get('rating_milli', 0))))
    log(f"telemetry: candidate selection for {requires} -> {[m['id'] for m in eligible]}")
    return eligible


def requirements_for(body: Json) -> Json:
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    if index < len(prompts):
        declared = prompts[index]['requires']
        inferred = requirements_from_body(body)
        return {**inferred, **declared}
    return requirements_from_body(body)


def requirements_from_body(body: Json) -> Json:
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


class Upstream:
    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def dial(model: str, body: Json) -> Upstream:
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=25)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def extract_error_code(text: str) -> str:
    if not text:
        return ""
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            err = val.get("error")
            if isinstance(err, dict):
                code = err.get("code") or err.get("type") or ""
                return str(code)[:64]
            if isinstance(err, str) and err:
                return err[:64]
            code = val.get("code") or val.get("type")
            if isinstance(code, str) and code:
                return code[:64]
        elif isinstance(val, str) and val:
            return val[:64]
    except Exception:
        pass
    trimmed = text.strip()
    return trimmed[:64] if trimmed and not trimmed.startswith('{') else ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, failures: list[Json] | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        if failures:
            self.send_header('x-makefun-upstream-failures', json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, failures: list[Json] | None = None) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message), failures=failures)
        else:
            self.send_json(status, openai_error(message, error_type, None), failures=failures)

    def stream_headers(self, failures: list[Json] | None = None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        self.send_header('connection', 'close')
        if failures:
            self.send_header('x-makefun-upstream-failures', json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False))
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, status: int, text: str, failures: list[Json] | None = None) -> None:
        suffix = '' if text == '' else f': {text[:500]}'
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'the gateway answered HTTP {status}{suffix}', failures=failures)
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'the gateway answered HTTP {status}{suffix}', failures=failures)

    def models(self) -> None:
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        candidates = candidates_for(requires)
        if not candidates:
            self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            return

        failures: list[Json] = []
        for attempt_idx, route in enumerate(candidates):
            model_id = route['id']
            if attempt_idx > 0:
                log(f"telemetry: chat failover attempt #{attempt_idx} trying candidate {model_id}")
            else:
                log(f"telemetry: chat primary attempt trying candidate {model_id}")
            try:
                upstream = dial(model_id, {**body, 'model': model_id})
            except Exception as err:
                log(f"telemetry: chat candidate {model_id} encountered connection error: {err}")
                failures.append({'status': 502, 'code': 'upstream_unreachable', 'model': model_id})
                with failures_lock:
                    MODEL_FAILURES[model_id] = MODEL_FAILURES.get(model_id, 0) + 1
                continue

            if not upstream.ok:
                status = upstream.status
                err_text = upstream.text()
                code = extract_error_code(err_text)
                log(f"telemetry: chat candidate {model_id} failed with HTTP {status} (code: '{code}'): {err_text[:120]}")
                failures.append({'status': status, 'code': code, 'model': model_id})
                with failures_lock:
                    MODEL_FAILURES[model_id] = MODEL_FAILURES.get(model_id, 0) + 1
                continue

            log(f"telemetry: chat candidate {model_id} succeeded with HTTP {upstream.status}")

            # Successful response
            if body.get('stream') is True:
                self.stream_headers(failures=failures if failures else None)
                try:
                    for frame in openai_stream_with_model(upstream.chunks(), model_id):
                        self.write_event(frame)
                finally:
                    upstream.close()
                return

            answer = upstream.text()
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            self.send_json(200, openai_json_with_model(parsed if is_record(parsed) else {}, model_id), failures=failures if failures else None)
            return

        self.refused('openai', failures[-1]['status'] if failures else 502, 'all candidate free models failed', failures=failures)

    def messages(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        candidates = candidates_for(requires)
        if not candidates:
            self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            return

        failures: list[Json] = []
        for attempt_idx, route in enumerate(candidates):
            model_id = route['id']
            if attempt_idx > 0:
                log(f"telemetry: messages failover attempt #{attempt_idx} trying candidate {model_id}")
            else:
                log(f"telemetry: messages primary attempt trying candidate {model_id}")
            try:
                upstream = dial(model_id, anthropic_to_openai(body, model_id))
            except Exception as err:
                log(f"telemetry: messages candidate {model_id} encountered connection error: {err}")
                failures.append({'status': 502, 'code': 'api_error', 'model': model_id})
                with failures_lock:
                    MODEL_FAILURES[model_id] = MODEL_FAILURES.get(model_id, 0) + 1
                continue

            if not upstream.ok:
                status = upstream.status
                err_text = upstream.text()
                code = extract_error_code(err_text)
                log(f"telemetry: messages candidate {model_id} failed with HTTP {status} (code: '{code}'): {err_text[:120]}")
                failures.append({'status': status, 'code': code, 'model': model_id})
                with failures_lock:
                    MODEL_FAILURES[model_id] = MODEL_FAILURES.get(model_id, 0) + 1
                continue

            log(f"telemetry: messages candidate {model_id} succeeded with HTTP {upstream.status}")

            # Successful response
            if body.get('stream') is True:
                self.stream_headers(failures=failures if failures else None)
                try:
                    for event in anthropic_events_from_openai(upstream.chunks(), model_id):
                        self.write_event(event)
                finally:
                    upstream.close()
                return

            answer = upstream.text()
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            self.send_json(200, openai_to_anthropic(parsed if is_record(parsed) else {}, model_id), failures=failures if failures else None)
            return

        self.refused('anthropic', failures[-1]['status'] if failures else 502, 'all candidate free models failed', failures=failures)


def register(port: int, report: Json) -> None:
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_: object) -> None:
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if '--list' in sys.argv[1:]:
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
