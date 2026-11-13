#!/usr/bin/env python3
"""Linux x86-64 live #588 HTTP/model probe (requires gdb and local Qwen3 weights).

Observe the unmodified production tokenizer's ChatML argument with a read-only
breakpoint. This verifies routing independently of probabilistic language output.
All requests still run real model inference. No reporter-corpus quality claim.

Example (unstripped CPU build):
  python tools/diagnostics/server_context_http.py --binary build/bin/crispasr \
    --model "$CRISPASR_MODELS_DIR/qwen3-asr-0.6b-q4_k.gguf" --output-dir "$TMPDIR/context-http"
The output directory contains full prompts and responses; use public fixtures.
"""
import argparse
import hashlib
import http.client
import io
import json
import os
import platform
import re
from pathlib import Path
import signal
import socket
import subprocess
import time
import wave


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audio', type=Path, default=Path(__file__).resolve().parents[2] / 'samples/jfk.wav')
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("the read-only register observer requires Linux x86-64")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    traces = out / 'tokenizer.jsonl'
    traces.write_text('')
    # Small real speech fixture, retaining the original PCM encoding.
    with wave.open(str(args.audio), 'rb') as source:
        payload = io.BytesIO()
        with wave.open(payload, 'wb') as target:
            target.setparams(source.getparams())
            target.writeframes(source.readframes(source.getframerate() * 4))
    audio = payload.getvalue()
    (out / 'input.wav').write_bytes(audio)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    gdb_script = out / 'observe.gdb'
    gdb_script.write_text('''set pagination off
set confirm off
set breakpoint pending on
python
import gdb, json
class PromptObserver(gdb.Breakpoint):
    def stop(self):
        # SysV x86-64 ABI: tokenizer(ctx, const char *text, int *n_tokens).
        text = gdb.parse_and_eval('(char *)$rsi').string(encoding='utf-8')
        with open(TRACE_PATH, 'a') as stream:
            stream.write(json.dumps({'prompt': text}, ensure_ascii=False) + '\\n')
        return False
PromptObserver('qwen3_asr_tokenize')
end
run
'''.replace('TRACE_PATH', repr(str(traces))))
    command = ['gdb', '--batch', '-x', str(gdb_script), '--args', str(args.binary.resolve()),
               '--server', '--no-warmup', '--backend', 'qwen3', '-m', str(args.model.resolve()),
               '--gpu-backend', 'cpu', '--no-gpu', '--lid-backend', 'off',
               '--host', '127.0.0.1', '--port', str(port), '--context', 'English', '-t', '2']
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    env.pop('CRISPASR_API_KEYS', None)
    cases = [
        ('default', '/v1/audio/transcriptions', {}, 'English'),
        ('override', '/v1/audio/transcriptions', {'context': 'English\nРусский: verbatim.'}, 'English\nРусский: verbatim.'),
        ('restore', '/v1/audio/transcriptions', {}, 'English'),
        ('empty', '/v1/audio/transcriptions', {'context': ''}, 'English'),
        ('prompt-independent', '/v1/audio/transcriptions', {'prompt': 'UNRELATED_DECODER_PREFIX'}, 'English'),
        ('native-override', '/inference', {'context': 'English speech'}, 'English speech'),
        ('native-restore', '/inference', {}, 'English'),
        ('stream-override', '/v1/audio/transcriptions', {'context': 'English speech', 'stream': 'true'}, 'English speech'),
        ('stream-restore', '/v1/audio/transcriptions', {'stream': 'true'}, 'English'),
    ]
    results = []
    with (out / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited before readiness; inspect server.log')
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    ready = response.status == 200
                    response.read()
                    connection.close()
                    if ready:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server readiness')
                time.sleep(0.25)
            for name, endpoint, fields, expected in cases:
                boundary = 'crispasr-context-live'
                fields = {'response_format': 'json', 'max_tokens': '48', **fields}
                body = b''
                for key, value in fields.items():
                    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n').encode()
                body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode() + audio
                body += f'\r\n--{boundary}--\r\n'.encode()
                started = time.monotonic()
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
                connection.request('POST', endpoint, body, {'Content-Type': f'multipart/form-data; boundary={boundary}'})
                response = connection.getresponse()
                raw = response.read().decode('utf-8')
                status = response.status
                content_type = response.getheader('Content-Type', '')
                connection.close()
                (out / f'{name}.response').write_text(raw)
                assert status == 200, (name, status, raw)
                if content_type.startswith('text/event-stream'):
                    response_mode = 'sse'
                    assert fields.get('stream') == 'true'
                    assert 'data: [DONE]' in raw, (name, raw)
                    text = ''.join(json.loads(line[6:])['text'] for line in raw.splitlines()
                                   if line.startswith('data: {'))
                else:
                    # Qwen3 currently omits CAP_STREAMING: stream=true uses
                    # the production buffered fallback, which still needs context.
                    response_mode = 'buffered-stream-fallback' if fields.get('stream') == 'true' else 'buffered'
                    text = json.loads(raw)['text']
                assert len(text.split()) >= 3, (name, 'missing real model output', text)
                if name in {'default', 'restore', 'empty', 'prompt-independent', 'native-restore', 'stream-restore'} and args.audio.resolve() == (Path(__file__).resolve().parents[2] / 'samples/jfk.wav').resolve():
                    words = set(re.findall(r'[a-z]+', text.lower()))
                    assert len(words & {'and', 'so', 'my', 'fellow', 'americans'}) >= 4, (name, 'speech control', text)
                # The adapter also tokenizes audio-pad and end-token markers.
                prompts = [value for line in traces.read_text().splitlines()
                           if (value := json.loads(line)['prompt']).startswith('<|im_start|>system\n')]
                assert len(prompts) == len(results) + 1, (name, 'unexpected tokenizer count', len(prompts))
                actual = prompts[-1]
                prefix = '<|im_start|>system\n' + expected + '<|im_end|>\n'
                assert actual.startswith(prefix), (name, 'wrong context', actual[:200])
                assert actual.endswith('<|im_start|>assistant\n'), (name, 'unexpected forced-language prefill')
                assert 'UNRELATED_DECODER_PREFIX' not in actual, (name, 'prompt became context')
                results.append({'case': name, 'endpoint': endpoint, 'context': expected,
                                'status': status, 'text': text, 'response_mode': response_mode, 'prompt_sha256': hashlib.sha256(actual.encode()).hexdigest(),
                                'elapsed_s': round(time.monotonic() - started, 3)})
                print(name + ': PASS ' + text, flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    receipt = {'test_source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'binary_version': subprocess.check_output([str(args.binary.resolve()), '--version'], text=True, stderr=subprocess.STDOUT).strip(),
               'binary_sha256': sha(args.binary), 'model_sha256': sha(args.model),
               'input_sha256': hashlib.sha256(audio).hexdigest(), 'device': 'CPU',
               'method': 'Unmodified server; gdb read-only tokenizer argument observation; real multipart HTTP and model inference',
               'cases': results, 'passed': len(results) == len(cases),
               'limits': 'One four-second English fixture; no reporter ru/en corpus or language-quality qualification. Qwen3 currently omits CAP_STREAMING, so stream=true exercises buffered fallback, not genuine SSE.'}
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
