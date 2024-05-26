import pixeltable as pxt
import numpy as np
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import ComponentIterator
from typing import Any, Iterator
import spacy
import av


@pxt.expr_udf
def e5_embed(text: str) -> np.ndarray:
    return sentence_transformer(text, model_id='intfloat/e5-large-v2')

def extract_audio(video_path: str, audio_path: str, stream_index: int = 0):
    # remuxes audio into its own container file
    # https://pyav.basswood-io.com/docs/stable/cookbook/basics.html#remuxing

    input_ = av.open(video_path, "r")
    output = av.open("output.", "w")

    # Make an output stream using the input as a template. This copies the stream
    # setup from one to the other.
    try:
        in_stream = input_.streams.audio[0]
        out_stream = output.add_stream(template=in_stream)

        for packet in input_.demux(in_stream):
            # We need to skip the "flushing" packets that `demux` generates.
            if packet.dts is None:
                continue

            # We need to assign the packet to the new stream.
            packet.stream = out_stream
            output.mux(packet)
    finally:
        input_.close()
        output.close()


class AudioSplitter(ComponentIterator):
    def __init__(self, audio_file: str, max_length_bytes: int):
        self.audio_file = audio_file
        self.max_length_bytes = max_length_bytes
        self.iter = self._iter()

    def _iter(self) -> Iterator[dict[str, Any]]:
        for sentence in self.doc.sents:
            yield {'text': sentence.text}

    def __next__(self) -> dict[str, Any]:
        return next(self.iter)

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        pass

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, pxt.ColumnType]:
        return {'text': pxt.StringType()}

    @classmethod
    def output_schema(cls,  *args: Any, **kwargs: Any) -> tuple[dict[str, pxt.ColumnType], list[str]]:
        return {'text': pxt.StringType()}, []


class SentenceSplitter(ComponentIterator):
    def __init__(self, text: str):
        self._text = text
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.doc = self.spacy_nlp(self._text)
        self.iter = self._iter()

    def _iter(self) -> Iterator[dict[str, Any]]:
        for sentence in self.doc.sents:
            yield {'text': sentence.text}

    def __next__(self) -> dict[str, Any]:
        return next(self.iter)

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        pass

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, pxt.ColumnType]:
        return {'text': pxt.StringType()}

    @classmethod
    def output_schema(cls,  *args: Any, **kwargs: Any) -> tuple[dict[str, pxt.ColumnType], list[str]]:
        return {'text': pxt.StringType()}, []