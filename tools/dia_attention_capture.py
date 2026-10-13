"""Owned snapshots of the actual pinned original self-attention calls.

This observes modules and caches; it does not implement attention or replace
any original computation. Arrays keep (batch, time, heads, head_dim) order.
"""


def parse_attention_layers(value):
    if value == "":
        return ()
    parts = value.split(",")
    if any(not p.isascii() or not p.isdecimal() for p in parts):
        raise ValueError("DIA_CAPTURE_ATTENTION_LAYERS must be comma-separated decimal indices")
    layers = tuple(int(p) for p in parts)
    if len(set(layers)) != len(layers):
        raise ValueError("DIA_CAPTURE_ATTENTION_LAYERS contains duplicate indices")
    return layers


def validate_attention_layers(layers, count):
    if any(type(i) is not int or not 0 <= i < count for i in layers) or len(set(layers)) != len(layers):
        raise ValueError("Invalid or duplicate Dia attention layer index")
    return tuple(layers)


def attention_stages(layers):
    return [f"decoder_sa{i}_{part}" for i in layers for part in ("q", "k", "v", "attention")]


def install_attention_hooks(decoder_layers, selected, step, active, save):
    handles = []
    try:
        for index in selected:
            module = decoder_layers[index].self_attention
            prefix = f"decoder_sa{index}_"
            observed = {"rotations": 0}

            def begin(m, args, kwargs, observed=observed):
                if active():
                    positions = kwargs.get("q_positions", args[2] if len(args) > 2 else None)
                    if positions is None or tuple(positions.shape) != (2, 1) or not bool((positions == step).all()):
                        raise ValueError("Dia attention capture position is not the requested original call")
                    observed["rotations"] = 0

            def rotate(_m, _args, output, module=module, prefix=prefix, observed=observed):
                if active():
                    slot = observed["rotations"]
                    if slot > 1:
                        raise ValueError("Unexpected extra original self-attention rotation")
                    heads = module.num_query_heads if slot == 0 else module.num_kv_heads
                    if tuple(output.shape) != (2, 1, heads, module.head_dim):
                        raise ValueError("Unexpected original self-attention Q/K shape")
                    save(prefix + ("q" if slot == 0 else "rotated_current_k"), output)
                    observed["rotations"] += 1

            def before_projection(_m, args, module=module, prefix=prefix):
                if active():
                    value = args[0]
                    if tuple(value.shape) != (2, 1, module.num_query_heads, module.head_dim):
                        raise ValueError("Unexpected original pre-output-projection attention shape")
                    save(prefix + "attention", value)

            def finish(m, args, kwargs, output, prefix=prefix, observed=observed):
                if active() and output is not None:
                    if observed["rotations"] != 2:
                        raise ValueError("Original self-attention did not rotate both Q and K")
                    cache = kwargs.get("cache", args[5] if len(args) > 5 else None)
                    if cache is None or int(cache.current_idx) != step + 1:
                        raise ValueError("Unexpected original self-attention cache history length")
                    shape = (2, m.num_kv_heads, step + 1, m.head_dim)
                    for name in ("k", "v"):
                        value = getattr(cache, name)[:, :, :step + 1]
                        if tuple(value.shape) != shape:
                            raise ValueError("Unexpected original self-attention cache shape")
                        save(prefix + name, value.transpose(1, 2))

            handles.append(module.register_forward_pre_hook(begin, with_kwargs=True))
            handles.append(module.rotary_emb.register_forward_hook(rotate))
            handles.append(module.o_proj.register_forward_pre_hook(before_projection))
            handles.append(module.register_forward_hook(finish, with_kwargs=True, always_call=True))
        return handles
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
