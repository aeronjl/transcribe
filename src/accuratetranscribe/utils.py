"""This module provides utilities for transcription."""

import datetime
import os
from io import BytesIO
from typing import Optional, Generator, Union
from contextlib import contextmanager
import tempfile

from pydub import AudioSegment


@contextmanager
def temporary_file(suffix: Optional[str] = None) -> Generator[str, None, None]:
    """Context manager for creating temporary files."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.close()
        yield temp_file.name
    finally:
        os.unlink(temp_file.name)


def convert_to_timestamp(seconds: Union[int, float]) -> str:
    """Convert seconds to HH:MM:SS format."""
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    timestamp = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return timestamp


def buffer_audio(audio: AudioSegment) -> BytesIO:
    """Convert AudioSegment to BytesIO buffer."""
    buffer = BytesIO()
    buffer.name = "buffer.wav"
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer


# Main execution
if __name__ == "__main__":
    pass

