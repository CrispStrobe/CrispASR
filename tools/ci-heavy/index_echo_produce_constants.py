"""Pinned source shared by Index-Echo conversion and benchmark jobs."""
SOURCE = 'IndexTeam/Index-Echo-S2TT-2B'
REVISION = '5d98a34d9685869b11e9c94d01dee22a8b8e53b5'
LLAMA_REVISION = '42d958167a748f2c04b1f888e84e7a58f609ddcb'
DESTINATION = 'cstr/index-echo-2b-GGUF'

MODELS = {
    '2b': (SOURCE, REVISION, DESTINATION),
    '9b': ('IndexTeam/Index-Echo-S2TT-9B', 'b8ac6fb7d3dc17cee48a52201bd3d93dc86b0dba',
           'cstr/index-echo-9b-GGUF'),
}
LICENSE_URL = 'https://raw.githubusercontent.com/bilibili/Index-Translate/8168c799051b4180c32c4c0469d255e80fdb2b2d/LICENSE'
