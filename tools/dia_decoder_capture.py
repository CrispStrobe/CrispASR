"""Observe actual original decoder calls without replacing their arithmetic."""

from dia_attention_capture import parse_attention_layers, validate_attention_layers

PARTS = ("input", "pre_sa_norm", "sa_attention", "sa_output", "sa_residual",
         "pre_ca_norm", "cross_q", "cross_attention", "cross_output", "ca_residual",
         "pre_mlp_norm", "mlp_gate", "mlp_up", "mlp_hidden", "mlp_output", "output")


def parse_boundary_layers(value):
    try:
        return parse_attention_layers(value)
    except ValueError as error:
        raise ValueError(str(error).replace("DIA_CAPTURE_ATTENTION_LAYERS", "DIA_CAPTURE_BOUNDARY_LAYERS")) from error


def boundary_stages(layers):
    return [f"decoder_block{i}_{part}" for i in layers for part in PARTS]


def install_boundary_hooks(decoder_layers, selected, step, active, save):
    """Read residuals from actual norm inputs and hidden from actual wo input."""
    selected = validate_attention_layers(selected, len(decoder_layers))
    handles = []
    try:
        for index in selected:
            layer = decoder_layers[index]
            prefix = f"decoder_block{index}_"
            width = layer.pre_sa_norm.weight.numel()
            intermediate = layer.mlp.wi_fused.out_features[1]

            def observe(part, shape, prefix=prefix):
                def record(value):
                    if active():
                        if tuple(value.shape) != shape:
                            raise ValueError(f"Unexpected original decoder boundary shape: {prefix}{part}")
                        save(prefix + part, value)
                return record

            def input_hook(record):
                return lambda _m, args: record(args[0])

            def output_hook(record):
                return lambda _m, _args, output: record(output)

            def begin(_m, args, kwargs, prefix=prefix):
                if active():
                    state = kwargs.get("state", args[1] if len(args) > 1 else None)
                    positions = None if state is None else state.dec_positions
                    if positions is None or tuple(positions.shape) != (2, 1) or not bool((positions == step).all()):
                        raise ValueError(f"Original decoder boundary position mismatch: {prefix}")

            handles.append(layer.register_forward_pre_hook(begin, with_kwargs=True))
            vector = (2, 1, width)
            for module, part in ((layer.pre_sa_norm, "input"), (layer.pre_ca_norm, "sa_residual"),
                                 (layer.pre_mlp_norm, "ca_residual")):
                handles.append(module.register_forward_pre_hook(input_hook(observe(part, vector))))
            for module, part in ((layer.pre_sa_norm, "pre_sa_norm"), (layer.self_attention, "sa_output"),
                                 (layer.pre_ca_norm, "pre_ca_norm"), (layer.cross_attention, "cross_output"),
                                 (layer.pre_mlp_norm, "pre_mlp_norm"), (layer.mlp, "mlp_output"), (layer, "output")):
                handles.append(module.register_forward_hook(output_hook(observe(part, vector))))
            for module, part in ((layer.self_attention, "sa_attention"), (layer.cross_attention, "cross_attention")):
                shape = (2, 1, module.num_query_heads, module.head_dim)
                handles.append(module.o_proj.register_forward_pre_hook(input_hook(observe(part, shape))))
            ca = layer.cross_attention
            handles.append(ca.rotary_emb.register_forward_hook(
                output_hook(observe("cross_q", (2, 1, ca.num_query_heads, ca.head_dim)))))
            gate = observe("mlp_gate", (2, 1, intermediate))
            up = observe("mlp_up", (2, 1, intermediate))

            def fused(_m, _args, output, gate=gate, up=up, intermediate=intermediate):
                if active():
                    if tuple(output.shape) != (2, 1, 2, intermediate):
                        raise ValueError("Unexpected original fused MLP projection shape")
                    gate(output[..., 0, :])
                    up(output[..., 1, :])

            handles.append(layer.mlp.wi_fused.register_forward_hook(fused))
            handles.append(layer.mlp.wo.register_forward_pre_hook(
                input_hook(observe("mlp_hidden", (2, 1, intermediate)))))
        return handles
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
