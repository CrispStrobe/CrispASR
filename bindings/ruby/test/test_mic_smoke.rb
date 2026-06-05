require_relative "helper"

# Smoke test for the Ruby microphone binding (PLAN #62/#62d).
#
# Opens a real capture device, runs ~1s, and asserts the audio thread
# fired the user block at least 10 times. Catches the common GVL/MRI
# failure modes that otherwise crash CI silently:
#   - Audio thread reaching into MRI without the GVL → segfault
#   - User Proc GC'd while the device is running → segfault on dispatch
#   - close() not joining the pump thread → use-after-free on the queue
#
# Gated behind CRISPASR_MIC_TEST=1 because CI runners typically have
# no microphone and the device-open call would fail.
class TestMicSmoke < TestBase
  def setup
    omit "set CRISPASR_MIC_TEST=1 to run mic tests" unless ENV["CRISPASR_MIC_TEST"] == "1"
  end

  def test_mic_fires_callbacks
    mu = Mutex.new
    callbacks = 0
    total_samples = 0

    handle = Whisper::CrispASR::Mic.open(16000, 1) do |pcm|
      mu.synchronize do
        callbacks += 1
        total_samples += pcm.size
      end
    end

    begin
      Whisper::CrispASR::Mic.start(handle)
      sleep 1.0
      Whisper::CrispASR::Mic.stop(handle)
    ensure
      Whisper::CrispASR::Mic.close(handle)
    end

    cb_count = mu.synchronize { callbacks }
    samples  = mu.synchronize { total_samples }

    # 16 kHz mono with miniaudio's 256-4096 sample buffers ⇒ ~4-60
    # callbacks/s. 10 is a safe floor.
    assert cb_count >= 10, "expected ≥10 mic callbacks in 1s, got #{cb_count}"
    assert samples  > 8000, "expected ≥0.5s of audio, got #{samples} samples"
  end

  def test_default_device_name
    name = Whisper::CrispASR::Mic.default_device_name
    assert_not_nil name
    # Empty is allowed (headless host with no input device); we just
    # want to confirm the symbol resolves without crashing.
    puts "default capture device: '#{name}'"
  end
end
