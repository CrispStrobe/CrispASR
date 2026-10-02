"""Predeclared Index-Echo decoder Q4_K experiments; not default quant rules."""

RECIPES = {
    'q4_k_plain': [],
    'q4_k_sensitive': [r'^(token_embd|output)\.weight$=f16',
        r'^blk\.[0-9]+\.ssm_=f16', r'^blk\.[0-9]+\.attn_=q8_0',
        r'^blk\.[0-9]+\.ffn_down\.weight$=q8_0'],
    'q4_k_ffn_guarded': [r'^blk\.[0-9]+\.ffn_(gate|up)\.weight$=q4_k',
        r'^blk\.[0-9]+\.ffn_down\.weight$=q8_0', r'.*=f16'],
    'q4_k_middle': [r'^blk\.([4-9]|1[0-9]|2[0-7])\.ffn_(gate|up)\.weight$=q4_k',
        r'.*=f16'],
}


def audit(tensors, name):
    """Audit physical types, including retained sensitive and small tensors."""
    types = {t['name']: t['type'] for t in tensors}
    q4_bytes = sum(t['bytes'] for t in tensors if t['type'] == 'Q4_K')
    assert q4_bytes > 0, 'No physical Q4_K matrices'
    if name != 'q4_k_plain':
        assert types['token_embd.weight'] == types['output.weight'] == 'F16'
        assert all(t['type'] in ('F16', 'F32') for t in tensors if '.ssm_' in t['name'])
    if name == 'q4_k_ffn_guarded':
        assert sum(t['type'] == 'Q4_K' for t in tensors) == 64
        assert sum(t['type'] == 'Q8_0' for t in tensors) == 32
    if name == 'q4_k_middle':
        assert sum(t['type'] == 'Q4_K' for t in tensors) == 48
        assert not any(t['type'] == 'Q8_0' for t in tensors)
    return q4_bytes
