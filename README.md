Transcription with OpenAI's [Whisper](https://github.com/openai/whisper) is very accurate, but it doesn't natively support speaker labelling (diarisation). 

Existing libraries for diarisation like [pyannote](https://github.com/pyannote/pyannote-audio) rely on audio features to separate and identify speakers, but are computationally expensive and often inaccurate. A common failure mode arises when the speaker changes their audio quality, such as when they move closer to or further from the microphone. This can cause the diarisation algorithm to incorrectly identify the speaker as a new person.

I had a simple hypothesis: the cues from transcribed speech are sufficient to identify speakers. I developed a pipeline which passes the transcribed text to GPT-4o with a prompt asking it to identify the speaker.

```mermaid
flowchart LR
    A[Input file #40;audio, video#41;]
    B[Whisper transcription]
    C[Text output]
    D[Label and tidy with GPT-4o]
    E[Output in user-defined format]
    A-- Convert to WAV -->B
    B-->C
    C-->D
    D-->C
    D-->E
```

# Installation

```bash
pip install precisetranscribe
```
