#!/usr/bin/env python3
"""Model-free checks for CSM step-hook counts, sequence shapes and cleanup."""

import importlib.util
from pathlib import Path
import sys


sys.dont_write_bytecode = True
script = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("csm_original_steps", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Handle:
    def __init__(self, callbacks, callback):
        self.callbacks, self.callback = callbacks, callback

    def remove(self):
        self.callbacks.remove(self.callback)


class FakeModule:
    def __init__(self):
        self.before, self.after = [], []

    def register_forward_pre_hook(self, callback, with_kwargs=False):
        assert with_kwargs
        self.before.append(callback)
        return Handle(self.before, callback)

    def register_forward_hook(self, callback, with_kwargs=False):
        assert with_kwargs
        self.after.append(callback)
        return Handle(self.after, callback)

    def __call__(self, value):
        for callback in self.before:
            callback(self, (value,), {})
        for callback in self.after:
            callback(self, (value,), {}, value)
        return value


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape

    def detach(self):
        return self


class FakeCuda:
    sync_count = 0

    def synchronize(self):
        self.sync_count += 1


class FakeTorch:
    cuda = FakeCuda()


class FakeModel:
    def __init__(self):
        self.backbone = FakeModule()
        self.decoder = FakeModule()

    def generate_frame(self, tokens):
        self.backbone(FakeTensor(1, tokens.shape[1], 2048))
        for index in range(31):
            self.decoder(FakeTensor(1, 2 if index == 0 else 1, 1024))
        return FakeTensor(1, 32)


model = FakeModel()
original = model.generate_frame.__func__
with module.FrameProfile(model, FakeTorch(), True) as profile:
    model.generate_frame(FakeTensor(1, 7, 33))
    model.generate_frame(FakeTensor(1, 1, 33))
module.validate_frames(profile.rows, accepted=1, eos_count=1, codebooks=32, timed=True)
summary = module.summarize_frames(profile.rows)
assert summary["prefill_backbone_forward_ms"] > 0
assert len(profile.samples) == 2 and FakeTorch.cuda.sync_count > 0
assert model.generate_frame.__func__ is original
assert not model.backbone.before and not model.backbone.after
assert not model.decoder.before and not model.decoder.after

with module.FrameProfile(model, FakeTorch(), False) as off:
    model.generate_frame(FakeTensor(1, 7, 33))
    model.generate_frame(FakeTensor(1, 1, 33))
module.validate_frames(off.rows, accepted=1, eos_count=1, codebooks=32, timed=False)
assert all(not row["decoder_calls"] for row in off.rows)

for bad_rows, accepted, eos in (
    (profile.rows, 2, 0),
    ([profile.rows[1], profile.rows[0]], 1, 1),
    ([profile.rows[0], {**profile.rows[1], "decoder_calls": []}], 1, 1),
):
    try:
        module.validate_frames(bad_rows, accepted, eos, 32, True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid CSM forward trace was accepted")

calls = [
    {"phase": phase, "pair": pair, "arm": arm, "attempted_frame_calls": 2,
     "accepted_audio_frames": 1, "all_attempted_codes_sha256": "a" * 64,
     "wav": {"sha256": "b" * 64}}
    for phase, pairs in (("warmup", 1), ("measured", module.MEASURED_PAIRS))
    for pair in range(pairs)
    for arm in ("timing_off", "timing_on")
]
module.validate_call_identity(calls)
for bad in (calls[:-1], [*calls[:-1], {**calls[-1], "all_attempted_codes_sha256": "c" * 64}],
            [*calls[:-1], {**calls[-1], "arm": "timing_off"}]):
    try:
        module.validate_call_identity(bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid timing-on/off identity or pair was accepted")

assert module.wer(module.PHRASE) == 0
print("CSM original step model-free controls: PASS")
