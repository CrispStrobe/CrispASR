#!/usr/bin/env python3
"""Hermetic actual-quantizer regression: YaRN embeddings and legacy/A-B policy."""
import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def fixture(path, arch, scaling):
    fields = [('general.architecture', 8, arch)]
    if scaling is not None:
        fields.append(('granite_speech.llm.rope_scaling.type', 8 if isinstance(scaling, str) else 4, scaling))
    names = ['token_embd.weight', 'blk.0.attn_q.weight']
    payload = b''.join(struct.pack('<e', math.sin(i * .13)) for i in range(256 * 4))
    with path.open('wb') as stream:
        stream.write(b'GGUF' + struct.pack('<IQQ', 3, len(names), len(fields)))
        for key, kind, value in fields:
            stream.write(string(key) + struct.pack('<I', kind))
            stream.write(string(value) if kind == 8 else struct.pack('<I', value))
        for index, name in enumerate(names):
            stream.write(string(name) + struct.pack('<IQQIQ', 2, 256, 4, 1, index * len(payload)))
        stream.write(b'\0' * (-stream.tell() % 32))
        stream.write(payload * len(names))
    return payload


def tensor_table(path):
    with path.open('rb') as stream:
        def number(fmt):
            return struct.unpack('<' + fmt, stream.read(struct.calcsize('<' + fmt)))[0]
        def text():
            return stream.read(number('Q')).decode()
        def value(kind):
            if kind == 8:
                return text()
            if kind == 9:
                subtype, count = number('I'), number('Q')
                return [value(subtype) for _ in range(count)]
            return number({0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
                           7: 'B', 10: 'Q', 11: 'q', 12: 'd'}[kind])
        assert stream.read(4) == b'GGUF' and number('I') == 3
        count, fields = number('Q'), number('Q')
        alignment = 32
        for _ in range(fields):
            key, kind = text(), number('I')
            item = value(kind)
            if key == 'general.alignment':
                alignment = item
        table = {}
        for _ in range(count):
            name = text()
            shape = [number('Q') for _ in range(number('I'))]
            kind, offset = number('I'), number('Q')
            table[name] = dict(shape=shape, type=kind, offset=offset)
        start = (stream.tell() + alignment - 1) // alignment * alignment
        for row in table.values():
            if row['type'] == 1:
                stream.seek(start + row['offset'])
                row['f16_bytes'] = stream.read(math.prod(row['shape']) * 2)
        return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('quantizer', type=Path)
    args = parser.parse_args()
    cases = [
        ('legacy', 'granite_speech', None, 'q8_0', [], 8),
        ('yarn', 'granite_speech', 'yarn', 'q8_0', [], 1),
        ('explicit-default', 'granite_speech', 'default', 'q8_0', [], 8),
        ('wrong-type', 'granite_speech', 1, 'q8_0', [], 8),
        ('other-architecture', 'whisper', 'yarn', 'q8_0', [], 8),
        ('legacy-nar', 'granite_nle', None, 'q8_0', [], 8),
        ('yarn-q4', 'granite_speech', 'yarn', 'q4_k', [], 1),
        ('override-old-policy', 'granite_speech', 'yarn', 'q8_0',
         ['--tensor-type', r'^token_embd\.weight$=q8_0'], 8),
    ]
    environment = {k: v for k, v in os.environ.items()
                   if k not in {'CRISPASR_GRANITE_QUANT_ALL', 'CRISPASR_GRANITE_ENC_F16'}}
    with tempfile.TemporaryDirectory(prefix='granite-quant-policy-') as temporary:
        root = Path(temporary)
        for label, arch, scaling, quant, extra, expected in cases:
            source, output = root / (label + '-in.gguf'), root / (label + '-out.gguf')
            original = fixture(source, arch, scaling)
            result = subprocess.run([str(args.quantizer.resolve()), str(source), str(output), quant, *extra],
                                    capture_output=True, text=True, env=environment, timeout=30)
            assert result.returncode == 0, (label, result.stdout, result.stderr)
            tensors = tensor_table(output)
            assert tensors['token_embd.weight']['type'] == expected, (label, tensors)
            assert tensors['blk.0.attn_q.weight']['type'] == (12 if quant == 'q4_k' else 8), label
            if expected == 1:
                assert tensors['token_embd.weight']['f16_bytes'] == original, label
            print('PASS', label)
    print('PASS: 8 actual quantizer cases; embedding bytes retained and legacy/A-B policy preserved')


if __name__ == '__main__':
    main()
