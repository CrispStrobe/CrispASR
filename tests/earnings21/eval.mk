PYTHON = python

CRISPASR_PREFIX = ../../
CRISPASR_MODEL = tiny

CRISPASR_CLI = $(CRISPASR_PREFIX)build/bin/crispasr
CRISPASR_FLAGS = --no-prints --language en --output-txt

# You can create eval.conf to override the CRISPASR_* variables
# defined above.
-include eval.conf

# Add  `EARNINGS21_EVAL10 = yes` to eval.conf to switch to a
# 10-hour subset. See "speech-datasets/earnings21/README.md" for
# more details about this subset.
ifdef EARNINGS21_EVAL10
METADATA_CSV = speech-datasets/earnings21/eval10-file-metadata.csv
AUDIO_SRCS = speech-datasets/earnings21/media/4320211.mp3 \
             speech-datasets/earnings21/media/4341191.mp3 \
             speech-datasets/earnings21/media/4346818.mp3 \
             speech-datasets/earnings21/media/4359971.mp3 \
             speech-datasets/earnings21/media/4365024.mp3 \
             speech-datasets/earnings21/media/4366522.mp3 \
             speech-datasets/earnings21/media/4366893.mp3 \
             speech-datasets/earnings21/media/4367535.mp3 \
             speech-datasets/earnings21/media/4383161.mp3 \
             speech-datasets/earnings21/media/4384964.mp3 \
             speech-datasets/earnings21/media/4387332.mp3
else
METADATA_CSV = speech-datasets/earnings21/earnings21-file-metadata.csv
AUDIO_SRCS = $(sort $(wildcard speech-datasets/earnings21/media/*.mp3))
endif

TRANS_TXTS = $(addsuffix .txt, $(AUDIO_SRCS))

# We output the evaluation result to this file.
DONE = $(CRISPASR_MODEL).txt

all: $(DONE)

$(DONE): $(TRANS_TXTS)
	$(PYTHON) eval.py $(METADATA_CSV) > $@.tmp
	mv $@.tmp $@

# Note: This task writes to a temporary file first to
# create the target file atomically.
%.mp3.txt: %.mp3
	$(CRISPASR_CLI) $(CRISPASR_FLAGS) --model $(CRISPASR_PREFIX)models/ggml-$(CRISPASR_MODEL).bin --file $^ --output-file $^.tmp
	mv $^.tmp.txt $^.txt

archive:
	tar -czf $(CRISPASR_MODEL).tar.gz --exclude="*.mp3" speech-datasets/earnings21/media $(DONE)

clean:
	@rm -f $(TRANS_TXTS)
	@rm -f $(DONE)

.PHONY: all archive clean
