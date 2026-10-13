"""Streaming original Dia neural precision controls; no reconstructed inference."""
import ast
import hashlib
from pathlib import Path

import numpy as np

from dia_dac_export import digest, mapped_gguf

WEIGHTS_SHA = 'caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26'
ROOT = Path(__file__).resolve().parents[1]


def parameter_catalog(keys):
    """Use the actual existing converter's name mapping verbatim."""
    path = ROOT / 'models/convert-dia-to-gguf.py'
    tree = ast.parse(path.read_text())
    names = {'map_encoder_tensor', 'map_decoder_tensor'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    result = {}
    for key in keys:
        family, separator, rest = key.partition('.')
        if not separator or family not in ('encoder', 'decoder'):
            raise ValueError(f'Unknown original neural tensor: {key}')
        name, transpose = namespace[f'map_{family}_tensor'](rest)
        if name is None:
            raise ValueError(f'Unmapped original neural tensor: {key}')
        if name.endswith('.wi_fused'):
            variants = [(name[:-len('.wi_fused')] + '.' + label, ('fused', index))
                        for index, label in enumerate(('gate', 'up'))]
        elif name == 'dia.decoder.logits_dense':
            variants = [(f'dia.decoder.heads.{index}', ('head', index)) for index in range(9)]
        else:
            variants = [(name, ('plain', transpose))]
        for output, transform in variants:
            if output in result:
                raise ValueError(f'Duplicate neural export: {output}')
            result[output] = (key, transform)
    return result


def original_array(name, catalog, get_tensor):
    key, (kind, option) = catalog[name]
    data = np.asarray(get_tensor(key))
    if data.dtype != np.float32 or not np.isfinite(data).all():
        raise ValueError(f'Original neural tensor must be finite F32: {key}')
    if kind == 'fused':
        if data.ndim != 3 or data.shape[1] != 2:
            raise ValueError(f'Unexpected original fused layout: {key}')
        data = data[:, option, :].T
    elif kind == 'head':
        if data.ndim != 3 or data.shape[1] != 9:
            raise ValueError(f'Unexpected original head layout: {key}')
        data = data[:, option, :].T
    elif option and data.ndim >= 2:
        if data.ndim == 3:
            data = (data.reshape(-1, data.shape[-1]).T if 'o_proj' in name
                    else data.reshape(data.shape[0], -1).T)
        else:
            data = data.T
    return np.ascontiguousarray(data)


def original_timescales(source, head_dim):
    """Invoke the actual pinned original nonpersistent-buffer constructor."""
    import torch
    from reference_backends import dia_tts_reference as reference
    source, hashes = reference.verify_source(source)
    if torch.__version__ != '2.6.0+cpu' or type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise ValueError('Original timescale export requires pinned Torch CPU and an even head dimension')
    path = source / 'dia/layers.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'RotaryEmbedding')
    namespace = dict(torch=torch, nn=torch.nn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    module = namespace['RotaryEmbedding'](head_dim, dtype=torch.float32)
    value = module.timescale.detach().cpu().numpy().copy()
    return value, dict(source_pin=reference.SOURCE_PIN, source_sha256=hashes['dia/layers.py'],
                      torch_version=str(torch.__version__), head_dim=head_dim)


def export_precision(keys, get_tensor, template, output, precision, checkpoint_sha=WEIGHTS_SHA,
                     timescales=None, timescale_metadata=None):
    """Promote original coefficients, preserving template schema and other bytes.

    The caller verifies the actual checkpoint hash before providing its tensors.
    GGUF's temporary spool prevents retaining a second full model in RAM.
    """
    from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType
    template, output = Path(template), Path(output)
    if checkpoint_sha != WEIGHTS_SHA:
        raise ValueError('Unexpected original neural checkpoint')
    if precision not in ('heads-f32', 'f32'):
        raise ValueError('Unknown neural precision policy')
    if output.resolve() == template.resolve() or output.exists():
        raise ValueError('Neural export requires a new output path')
    catalog = parameter_catalog(keys)
    auxiliary = []
    if timescales is not None:
        timescales = np.asarray(timescales)
        if (precision != 'f32' or timescales.dtype != np.float32 or timescales.ndim != 1
                or not timescales.size or not np.isfinite(timescales).all() or not (timescales > 0).all()
                or not timescale_metadata
                or timescale_metadata.get('source_pin') != '4a9e29b1bdbfe1be721353ec034a8c66c9f0a1a8'
                or timescale_metadata.get('source_sha256') != '9b77d590101783a7e570bd9889afb09b85912d05072313c32a984dbfe9323024'
                or timescale_metadata.get('torch_version') != '2.6.0+cpu'
                or timescale_metadata.get('head_dim') != 2 * timescales.size):
            raise ValueError('Invalid actual original timescale buffer or provenance')
        timescales = np.ascontiguousarray(timescales)
        auxiliary.append(dict(name='dia.rope.timescale', exported_type='F32', shape=list(timescales.shape),
                              sha256=hashlib.sha256(timescales.tobytes()).hexdigest(), source=timescale_metadata))
    with mapped_gguf(template) as reader:
        names = [tensor.name for tensor in reader.tensors]
        if len(names) != len(set(names)) or set(names) != set(catalog):
            raise ValueError('Original/template neural tensor coverage differs')
        if reader.fields['general.architecture'].contents() != 'dia':
            raise ValueError('Template architecture is not Dia')
        writer = GGUFWriter(str(output), 'dia', use_temp_file=True)
        audit = []
        try:
            for name, field in reader.fields.items():
                if name.startswith('GGUF.') or name == 'general.architecture':
                    continue
                subtype = field.types[1] if field.types[0] == GGUFValueType.ARRAY else None
                writer.add_key_value(name, field.contents(), field.types[0], subtype)
            if auxiliary:
                head = reader.fields.get('dia.attn_head_size')
                if head is None or head.contents() != 2 * timescales.size:
                    raise ValueError('Original timescale dimension differs from model')
                for key, value in timescale_metadata.items():
                    writer.add_string('dia.rope.timescale.' + key, str(value))
            for tensor in reader.tensors:
                if tensor.tensor_type not in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
                    raise ValueError(f'Unsupported template type: {tensor.name}')
                original = original_array(tensor.name, catalog, get_tensor)
                if original.shape != tensor.data.shape:
                    raise ValueError(f'Original/template shape differs: {tensor.name}')
                # The control must start from the actual rounded original model.
                if not np.array_equal(original.astype(tensor.data.dtype), tensor.data):
                    raise ValueError(f'Template is not rounded original coefficients: {tensor.name}')
                selected = precision == 'f32' or tensor.name.startswith('dia.decoder.heads.')
                value = original if selected else tensor.data
                writer.add_tensor(tensor.name, value)
                audit.append(dict(name=tensor.name, original_type=tensor.tensor_type.name,
                    exported_type='F32' if selected else tensor.tensor_type.name,
                    selected=selected, unchanged_bytes=value.tobytes() == tensor.data.tobytes(),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest()))
            if auxiliary:
                writer.add_tensor('dia.rope.timescale', timescales)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        with mapped_gguf(output) as exported:
            if [tensor.name for tensor in exported.tensors] != names + [row['name'] for row in auxiliary]:
                raise ValueError('Neural export changed tensor order')
            for tensor, expected in zip(exported.tensors, audit):
                if (tensor.tensor_type.name != expected['exported_type'] or
                    hashlib.sha256(tensor.data.tobytes()).hexdigest() != expected['sha256']):
                    raise ValueError(f'Physical neural export differs: {tensor.name}')
            if auxiliary:
                tensor = exported.tensors[-1]
                if tensor.tensor_type.name != 'F32' or not np.array_equal(tensor.data, timescales):
                    raise ValueError('Physical original timescale export differs')
            for name, field in reader.fields.items():
                if not name.startswith('GGUF.') and exported.fields[name].contents() != field.contents():
                    raise ValueError(f'Neural export changed metadata: {name}')
    return dict(precision=precision, original_weights_sha256=checkpoint_sha,
                mapping_source_sha256=digest(ROOT / 'models/convert-dia-to-gguf.py'),
                template_sha256=digest(template), export_sha256=digest(output),
                template_bytes=template.stat().st_size, export_bytes=output.stat().st_size,
                byte_delta=output.stat().st_size-template.stat().st_size, tensors=audit,
                auxiliary_tensors=auxiliary,
                production_defaults_changed=False, speech_acceptance=False)
