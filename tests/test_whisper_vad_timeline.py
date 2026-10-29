"""Literal VAD stitch/map/getters plus real decode entry code, without models.

Only VAD detections and the decoder body after entry bookkeeping are controlled;
no GGML inference or replacement timestamp implementation is compiled.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_session_utf8 import definition
from build_legacy_wrapper_probe import extract_function

ROOT = Path(__file__).resolve().parents[1]


class WhisperVadTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='whisper-vad-timeline-')
        path = Path(cls.tmp.name)
        source = (ROOT / 'src/crispasr.cpp').read_text()
        fixture = (ROOT / 'tests/fixtures/whisper-parallel-runtime.cpp').read_text().split('int main(')[0]
        fixture = fixture.replace('// PRODUCTION_VAD_TYPES', definition(source, 'struct vad_time_mapping {') + ';')
        fixture = fixture.replace('// PRODUCTION_VAD_SEGMENT_INFO', definition(source, 'struct vad_segment_info {') + ';')
        fixture = fixture.replace('// PRODUCTION_SEGMENT', definition(source, 'struct whisper_segment {') + ';')
        # Detector stubs precede the actual VAD stitcher, not a replacement map.
        vad_support = '\n'.join(definition(source, 'struct ' + name + ' {') + ';'
                                for name in ('whisper_vad_segment', 'whisper_vad_segments'))
        vad_support += '\n' + '\n'.join(definition(source, marker) for marker in (
            'static int cs_to_samples(', 'static int64_t samples_to_cs('))
        vad_support += r'''
struct whisper_vad_context {};
static bool fail_vad_init = false;
struct whisper_vad_context_params whisper_vad_default_context_params() { return {}; }
struct whisper_vad_context* whisper_vad_init_from_file_with_params(const char*, struct whisper_vad_context_params) {
    static whisper_vad_context ctx; return fail_vad_init ? nullptr : &ctx;
}
struct whisper_vad_segments* whisper_vad_segments_from_samples(whisper_vad_context*, whisper_vad_params, const float*, int) {
    ++vad_calls;
    if (vad_failure) return nullptr;
    auto* out = new whisper_vad_segments;
    if (!vad_silence) out->data = {{100,200},{500,600}};
    return out;
}
void whisper_vad_free_segments(whisper_vad_segments* segments) { delete segments; }
'''
        fixture = fixture.replace(definition(fixture, 'static bool whisper_vad('),
                                  vad_support + '\n' + definition(source, 'static bool whisper_vad('))
        fixture = fixture.replace(definition(fixture, 'int whisper_full('), definition(source, 'int whisper_full('))
        # Compile the literal public with_state entry before scheduler/inference.
        entry = definition(source, 'int whisper_full_with_state(').split('// Reset all schedulers')[0]
        entry = entry[entry.index('{') + 1:]
        fixture = fixture.replace('// PRODUCTION_DECODE_ENTRY', entry)
        reset_marker = 'static void whisper_reset_vad_mapping('
        if reset_marker in source:
            fixture = fixture.replace('// PRODUCTION_RESET_MAPPING', definition(source, reset_marker))
        fixture = fixture.replace('// PRODUCTION_PARALLEL', definition(source, 'int whisper_full_parallel('))
        for marker in ('static int64_t map_processed_to_original_time(',
                       'int64_t whisper_full_get_segment_t0_from_state(',
                       'int64_t whisper_full_get_segment_t1_from_state(',
                       'struct whisper_token_data whisper_full_get_token_data_from_state(',
                       'struct whisper_token_data whisper_full_get_token_data('):
            fixture += '\n' + definition(source, marker)
        api = (ROOT / 'src/crispasr_c_api.cpp').read_text()
        fixture += '\n#define CA_EXPORT extern "C"\n'
        for name in ('crispasr_token_t0', 'crispasr_token_t1', 'crispasr_token_dtw_t'):
            fixture += '\n' + extract_function(api, name)
        # Compile the real decoder seek arithmetic separately from GGML work.
        start = source.index('    const int seek_start = params.offset_ms / 10;')
        end = source.index('    // if length of spectrogram', start)
        fixture += '\nstatic int fixture_mel_length;\nint whisper_n_len_from_state(whisper_state*) { return fixture_mel_length; }\n'
        fixture += 'static std::pair<int, int> native_seek_range(whisper_full_params params, int length) {\n'
        fixture += 'fixture_mel_length = length; whisper_state* state = nullptr;\n' + source[start:end]
        fixture += 'return {seek_start, seek_end}; }\n'
        fixture += (ROOT / 'tests/fixtures/whisper-vad-timeline-main.cpp').read_text()
        (path / 'probe.cpp').write_text(fixture)
        cls.binary = path / 'vad-timeline-probe'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-pthread', '-O1',
                        '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
                        str(path / 'probe.cpp'), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_case(self, case):
        run = subprocess.run([str(self.binary), case], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_token_and_dtw_getters_map_original_time_without_mutating_storage(self):
        self.check_case('getters')

    def test_unavailable_times_and_no_mapping_keep_their_values(self):
        self.check_case('sentinels')

    def test_plain_sequential_call_clears_previous_vad_mapping(self):
        self.check_case('reuse-sequential')

    def test_plain_parallel_call_clears_previous_vad_mapping(self):
        self.check_case('reuse-parallel')

    def test_direct_state_call_clears_previous_vad_mapping(self):
        self.check_case('reuse-state')

    def test_real_stitcher_sequential_and_parallel_keep_original_token_times(self):
        self.check_case('stitch')

    def test_sequential_vad_detector_and_loader_failure_discard_old_results(self):
        self.check_case('failure')

    def test_vad_stitching_belongs_to_the_requested_state(self):
        self.check_case('state-owner')

    def test_sequential_decode_bounds_pcm_before_feature_extraction(self):
        self.check_case('bounded-sequential')

    def test_direct_state_decode_bounds_pcm_before_feature_extraction(self):
        self.check_case('bounded-state')

    def test_precomputed_mel_route_still_reaches_decoder(self):
        self.check_case('precomputed-mel')

    def test_seek_duration_never_extends_past_available_mel(self):
        self.check_case('seek-range')

    def test_empty_parallel_input_clears_previous_mapping(self):
        self.check_case('empty')

    def test_parallel_offset_past_end_clears_previous_mapping(self):
        self.check_case('offset-past-end')


if __name__ == '__main__':
    unittest.main()
