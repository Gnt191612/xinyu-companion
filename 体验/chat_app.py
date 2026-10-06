"""仅绑定本机的聊天体验入口，复用参赛推理与分类器。"""
import json
import os
import re
import sys
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'participant'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from auxiliary import AuxiliaryPredictor
from run_inference import load_config, model_path, parse_prediction, conversation_text, SYSTEM_PROMPT, EMOTIONS, PROFILE_LABELS, guard_listener_ack
from story_intent import assistant_story_requested, STORY_TASK

config = load_config()
experience_config_path = Path(os.environ.get('XINYU_CONFIG', str(Path(__file__).parent / 'config.json')))
experience_config = json.loads(experience_config_path.read_text(encoding='utf-8')) if experience_config_path.exists() else {}
path = ROOT / experience_config['model_path'] if experience_config.get('model_path') else model_path(config)
native_chat = experience_config.get('native_chat', False)
torch.set_num_threads(experience_config.get('cpu_threads', 4))
print('正在加载本机陪伴模型，请稍候……', flush=True)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
load_options = {'device_map': config['device_map'], 'torch_dtype': config['torch_dtype']}
if experience_config.get('load_in_4bit'):
    load_options.update(device_map={'': 'cuda'}, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **load_options)
ack_adapter = experience_config.get('ack_adapter_path')
adapter_path = experience_config.get('adapter_path')
if adapter_path or ack_adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ROOT / (adapter_path or ack_adapter), local_files_only=True)
model.eval()
classifier = AuxiliaryPredictor()
inference_lock = threading.Lock()
class CompleteResponse(StoppingCriteria):
    def __init__(self, prompt_length, prefix=''):
        self.prompt_length = prompt_length
        self.prefix = prefix

    def __call__(self, input_ids, scores, **kwargs):
        text = self.prefix + tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        match = re.search(r'"response_text"\s*:\s*', text)
        if not match:
            return False
        try:
            value, _ = json.JSONDecoder().raw_decode(text[match.end():])
            return isinstance(value, str) and bool(value.strip())
        except json.JSONDecodeError:
            return False

class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, content_type):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            status = {'service': 'xinyu-local', 'model_version': experience_config.get('model_version', '旧体验'),
                'loaded_in_4bit': bool(getattr(model, 'is_loaded_in_4bit', False)),
                'adapter_path': adapter_path, 'active_adapters': getattr(model, 'active_adapters', [])}
            self.send(200, json.dumps(status, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')
        elif self.path == '/':
            self.send(200, (Path(__file__).parent / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/motion.js':
            self.send(200, (Path(__file__).parent / 'motion.js').read_bytes(), 'text/javascript; charset=utf-8')
        elif self.path in ('/assets/xinyu-sakura.png', '/assets/sakura-panorama.png', '/assets/character.png', '/assets/scenery.png', '/assets/listening.png'):
            name = {'/assets/xinyu-sakura.png': '心语樱花.png', '/assets/sakura-panorama.png': '樱花全景.png', '/assets/character.png': '心语人物.png', '/assets/scenery.png': '樱花场景.png', '/assets/listening.png': '心语倾听.png'}[self.path]
            asset = Path(__file__).parent / 'assets' / name
            if asset.exists():
                self.send(200, asset.read_bytes(), 'image/png')
            else:
                self.send(404, b'', 'text/plain')
        else:
            self.send(404, b'', 'text/plain')

    def do_POST(self):
        if self.path != '/chat':
            self.send(404, b'', 'text/plain')
            return
        try:
            request_started = time.perf_counter()
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            history = data['history']
            if not history or len(history) > 40 or history[-1]['role'] != 'user':
                raise ValueError('请开启新对话后重试，最多支持20轮。')
            sample = {'id': '体验对话', 'history': history}
            companion_prompt = (ROOT / experience_config.get('companion_prompt_path', '体验/陪伴提示.txt')).read_text(encoding='utf-8')
            acknowledgment = bool(ack_adapter and len(history) >= 3 and re.fullmatch(
                r'(?:好的|好滴|嗯好|嗯嗯|好呀|好|可以|可以的|没问题|明白了|知道了|对[，, ]?就是这个意思)[。！!~～]*',
                history[-1]['content'].strip()))
            story_request = assistant_story_requested(history)
            native_reply = native_chat or acknowledgment
            if acknowledgment:
                companion_prompt = (Path(__file__).parent / '确认提示.txt').read_text(encoding='utf-8')
            if story_request:
                companion_prompt = '你是心语，一位温柔体贴的AI陪伴者。\n' + STORY_TASK.replace('只输出自然的故事回复，不输出JSON、规则或分析。', 'response_text只包含自然的故事回复，不包含规则或分析。')
            system_prompt = SYSTEM_PROMPT.format(
                emotions=', '.join(sorted(EMOTIONS)),
                personality=', '.join(sorted(PROFILE_LABELS['personality_traits'])),
                interests=', '.join(sorted(PROFILE_LABELS['interests'])),
                styles=', '.join(sorted(PROFILE_LABELS['style'])))
            messages = ([{'role': 'system', 'content': companion_prompt}, *history] if native_reply else
                        [{'role': 'system', 'content': system_prompt + '\n' + companion_prompt},
                         {'role': 'user', 'content': conversation_text(sample)}])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            story_prefix = '{"response_text":"从前，' if story_request else ''
            prompt += story_prefix
            inputs = tokenizer(prompt, return_tensors='pt').to(next(model.parameters()).device)
            if inputs['input_ids'].shape[1] > 12000:
                raise ValueError('对话过长，请开启新对话。')
            started = time.perf_counter()
            generation = {**config['generation'], 'max_new_tokens': 640 if story_request else (64 if acknowledgment else (192 if native_chat else 384)),
                          'repetition_penalty': 1.08 if native_reply else 1.05}
            if experience_config.get('response_only_generation') and not native_reply:
                generation['stopping_criteria'] = StoppingCriteriaList([CompleteResponse(inputs['input_ids'].shape[1], story_prefix)])
            adapter_scope = model.disable_adapter() if ack_adapter and not acknowledgment else nullcontext()
            with inference_lock, torch.inference_mode(), adapter_scope:
                generation_started = time.perf_counter()
                output = model.generate(**inputs, **generation)
                generation_finished = time.perf_counter()
            decoded = story_prefix + tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            if decoded.lstrip().startswith('{') or re.search(r'"response_text"\s*:', decoded):
                prediction, recovered = parse_prediction(sample['id'], decoded)
            elif re.match(r'^\s*response_text\s*[:：]', decoded):
                response = re.sub(r'^\s*response_text\s*[:：]\s*', '', decoded, count=1)
                response = re.split(r'\n\s*(?:emotion_label|user_profile|memory_refs)\s*[:：]', response, maxsplit=1)[0]
                prediction = {'response_text': response.strip()}
                recovered = True
            else:
                prediction = {'response_text': decoded.strip()}
                recovered = False
            if not prediction['response_text']:
                raise ValueError('本轮没有生成完整回复，请再试一次。')
            prediction['raw_response_text'] = prediction['response_text']
            prediction['raw_model_output'] = decoded
            prediction['response_text'], prediction['role_guard_applied'] = guard_listener_ack(history, prediction['response_text'])
            analysis_started = time.perf_counter()
            prediction['emotion_label'], prediction['user_profile'] = classifier.predict(sample)
            prediction.update(classifier.emotion_evidence(sample))
            prediction['story_mode'] = story_request
            prediction['seconds'] = round(time.perf_counter() - started, 2)
            prediction['timing'] = {'prepare_seconds': round(started - request_started, 3),
                'queue_seconds': round(generation_started - started, 3),
                'generation_seconds': round(generation_finished - generation_started, 3),
                'analysis_seconds': round(time.perf_counter() - analysis_started, 3),
                'total_seconds': round(time.perf_counter() - request_started, 3),
                'input_tokens': inputs['input_ids'].shape[1],
                'output_tokens': output.shape[1] - inputs['input_ids'].shape[1]}
            print(json.dumps({'event': 'chat_timing', **prediction['timing']}, ensure_ascii=False), flush=True)
            prediction['recovered'] = recovered
            self.send(200, json.dumps(prediction, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.send(400, json.dumps({'error': str(exc)}, ensure_ascii=False).encode(), 'application/json; charset=utf-8')


port = int(os.environ.get('XINYU_PORT', '7860'))
print(f'体验入口已就绪：http://127.0.0.1:{port}', flush=True)
ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
