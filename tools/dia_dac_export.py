"""Original-module Dia DAC precision export; no reconstructed neural math."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re

import numpy as np

OUTPUT_NAMES = {'decoder.snake1.alpha', 'decoder.conv2.weight', 'decoder.conv2.bias'}
CHECKPOINT_SHA = 'a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def parameter(codec, name):
    """Native loader names -> actual original modules, no reconstructed math."""
    match = re.fullmatch(r'quantizer\.quantizers\.(\d+)\.(codebook|in_proj|out_proj)\.(weight|bias)', name)
    if match:
        module = getattr(codec.quantizer.quantizers[int(match[1])], match[2])
        return getattr(module, match[3])
    if name.startswith('decoder.conv1.'):
        return getattr(codec.decoder.model[0], name.rsplit('.', 1)[1])
    if name.startswith('decoder.snake1.'):
        return getattr(codec.decoder.model[5], name.rsplit('.', 1)[1])
    if name.startswith('decoder.conv2.'):
        return getattr(codec.decoder.model[6], name.rsplit('.', 1)[1])
    match = re.fullmatch(r'decoder\.block\.(\d+)\.(snake1|conv_t1)\.(alpha|weight|bias)', name)
    if match:
        block = codec.decoder.model[int(match[1]) + 1].block
        return getattr(block[0 if match[2] == 'snake1' else 1], match[3])
    match = re.fullmatch(r'decoder\.block\.(\d+)\.res_unit([123])\.(snake1|conv1|snake2|conv2)\.(alpha|weight|bias)', name)
    if match:
        block = codec.decoder.model[int(match[1]) + 1].block[int(match[2]) + 1].block
        index = {'snake1': 0, 'conv1': 1, 'snake2': 2, 'conv2': 3}[match[3]]
        return getattr(block[index], match[4])
    raise ValueError(f'Unmapped published DAC tensor: {name}')


@contextmanager
def mapped_gguf(path):
    """Release reader mappings deterministically, including on CIFS errors."""
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    try:
        yield reader
    finally:
        # GGUFReader has no close/context API; its data is NumPy's memmap.
        reader.data._mmap.close()


def export_precision(codec, template, output, precision='output-f32', checkpoint_sha=CHECKPOINT_SHA):
    """Preserve template payloads except original F32 critical Parameters.

    Callers remove weight normalization with Torch's original utility first.
    The published rounded tensor cannot recover the original F32 coefficient.
    """
    from gguf import GGUFWriter, GGMLQuantizationType
    template, output = Path(template), Path(output)
    if output.resolve() == template.resolve() or output.exists():
        raise ValueError('Export requires a new output path')
    if precision not in ('output-f32', 'rvq-output-f32', 'f32'):
        raise ValueError('Unknown DAC precision policy')
    if checkpoint_sha != CHECKPOINT_SHA:
        raise ValueError('Unexpected original DAC checkpoint')
    with mapped_gguf(template) as reader:
        names = [tensor.name for tensor in reader.tensors]
        if len(names) != len(set(names)) or not OUTPUT_NAMES.issubset(names):
            raise ValueError('Incomplete or duplicate critical DAC tensors')
        payloads, promoted = [], []
        for tensor in reader.tensors:
            if tensor.tensor_type not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                raise ValueError(f'Unsupported published tensor type: {tensor.name}')
            original = parameter(codec, tensor.name).detach().cpu().float().numpy()
            if original.size != tensor.data.size:
                raise ValueError(f'Original tensor shape mismatch: {tensor.name}')
            selected = (precision == 'f32' or tensor.name in OUTPUT_NAMES or
                        precision == 'rvq-output-f32' and tensor.name.startswith('quantizer.'))
            value = original.reshape(tensor.data.shape) if selected else tensor.data
            value = np.ascontiguousarray(value).copy()
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite DAC tensor: {tensor.name}')
            payloads.append((tensor.name, value))
            if selected:
                promoted.append(tensor.name)
        # Match the already validated diagnostic archive schema and order.
        writer = GGUFWriter(str(output), 'dac-44khz')
        try:
            writer.add_string('crispasr.diagnostic.original_dac_sha256', checkpoint_sha)
            writer.add_string('crispasr.diagnostic.precision_variant', precision)
            for name, value in payloads:
                writer.add_tensor(name, value)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        with mapped_gguf(output) as exported:
            if [tensor.name for tensor in exported.tensors] != names:
                raise ValueError('Export changed tensor order or coverage')
            audit = []
            for old, new, (name, value) in zip(reader.tensors, exported.tensors, payloads):
                if new.data.shape != old.data.shape or new.data.dtype != value.dtype or new.data.tobytes() != value.tobytes():
                    raise ValueError(f'Exported payload mismatch: {name}')
                if name in promoted:
                    if new.tensor_type != GGMLQuantizationType.F32:
                        raise ValueError(f'Critical tensor is not F32: {name}')
                elif new.tensor_type != old.tensor_type or new.data.tobytes() != old.data.tobytes():
                    raise ValueError(f'Unrelated tensor changed: {name}')
                audit.append(dict(name=name, original_type=old.tensor_type.name, exported_type=new.tensor_type.name,
                                  promoted=name in promoted, unchanged_bytes=old.data.tobytes() == new.data.tobytes(),
                                  sha256=hashlib.sha256(new.data.tobytes()).hexdigest()))
            return dict(precision=precision, original_checkpoint_sha256=checkpoint_sha,
                        template_sha256=digest(template), export_sha256=digest(output),
                        template_bytes=template.stat().st_size, export_bytes=output.stat().st_size,
                        byte_delta=output.stat().st_size-template.stat().st_size, tensors=audit,
                        production_defaults_changed=False, speech_acceptance=False)
