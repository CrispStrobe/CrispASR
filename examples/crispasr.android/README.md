A sample Android app using [CrispASR](https://github.com/CrispStrobe/CrispASR) to do voice-to-text transcriptions.

To use:

1. Select a Whisper model from [ggerganov/whisper.cpp on Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main).[^1]
2. Copy the model to the "app/src/main/assets/models" folder.
3. Select a sample audio file (for example, [jfk.wav](https://raw.githubusercontent.com/CrispStrobe/CrispASR/main/samples/jfk.wav)).
4. Copy the sample to the "app/src/main/assets/samples" folder.
5. Select the "release" active build variant, and use Android Studio to run and deploy to your device.
[^1]: I recommend the tiny or base models for running on an Android device.

(PS: Do not move this android project folder individually to other folders, because this android project folder depends on the files of the whole project.)

<img width="300" alt="image" src="https://user-images.githubusercontent.com/1670775/221613663-a17bf770-27ef-45ab-9a46-a5f99ba65d2a.jpg">
