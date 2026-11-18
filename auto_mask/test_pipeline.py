"""Tests for pipeline logic that do not require either model service."""

from auto_mask.pipeline import parse_instruction_index, split_object_phrases


def test_instruction_index_mapping() -> None:
    assert parse_instruction_index(0) == ("seen", 0)
    assert parse_instruction_index(99) == ("seen", 99)
    assert parse_instruction_index(100) == ("unseen", 0)
    assert parse_instruction_index(199) == ("unseen", 99)


def test_instruction_index_rejects_out_of_range() -> None:
    for index in (-1, 200):
        try:
            parse_instruction_index(index)
        except ValueError:
            pass
        else:
            raise AssertionError(index)


def test_object_phrase_parser() -> None:
    assert split_object_phrases("the red block, the blue pad") == [
        "the red block",
        "the blue pad",
    ]
    assert split_object_phrases("<think>hidden</think>the bottle") == ["the bottle"]
    assert split_object_phrases("NONE") == []



def test_first_frame_reads_exactly_once(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from auto_mask.pipeline import first_frame_data_url

    calls = []
    class Capture:
        def isOpened(self):
            return True
        def read(self):
            calls.append('read')
            return True, object()
        def release(self):
            calls.append('release')
    cv2 = SimpleNamespace(VideoCapture=lambda path: Capture(), IMWRITE_JPEG_QUALITY=1,
                          imencode=lambda *args: (True, SimpleNamespace(tobytes=lambda: b'jpeg')))
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    assert first_frame_data_url(tmp_path / 'input.mp4') == 'data:image/jpeg;base64,anBlZw=='
    assert calls == ['read', 'release']


def test_qwen_payload_contains_one_image_and_no_video(monkeypatch):
    import io
    import json
    from auto_mask import pipeline

    requests = []
    def respond(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'robot, robot arm, bottle'},
                                                    'finish_reason': 'stop'}]}).encode())
    monkeypatch.setattr(pipeline.urllib.request, 'urlopen', respond)
    _, prompts = pipeline.extract_operation_objects('Lift the bottle.', api_url='http://localhost:8000',
                                                     first_frame_url='data:image/jpeg;base64,anBlZw==')
    assert prompts == ['robot', 'robot arm', 'bottle']
    req = requests[0]
    assert req.full_url == 'http://localhost:8000/v1/chat/completions'
    content = json.loads(req.data)['messages'][0]['content']
    assert [x['type'] for x in content] == ['text', 'image_url']
    assert content[1]['image_url']['url'] == 'data:image/jpeg;base64,anBlZw=='
    pipeline.extract_operation_objects('Lift the bottle.', api_url='http://localhost:8000/v1')
    assert isinstance(json.loads(requests[-1].data)['messages'][0]['content'], str)


def test_dataset_and_input_overwrite_rejected(tmp_path):
    import pytest
    from auto_mask.pipeline import DATASET_ROOT, validate_output_path

    source = tmp_path / 'source.mp4'
    with pytest.raises(ValueError):
        validate_output_path(source, source)
    with pytest.raises(ValueError):
        validate_output_path(source, DATASET_ROOT / 'any' / 'video' / 'episode0.mp4')
    validate_output_path(source, tmp_path / 'output.mp4')


def test_truncated_qwen_response_rejected(monkeypatch):
    import io
    import json
    import pytest
    from auto_mask import pipeline

    monkeypatch.setattr(pipeline.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(
        json.dumps({'choices': [{'message': {'content': 'robot, robot arm, white'},
                                'finish_reason': 'length'}]}).encode()))
    with pytest.raises(RuntimeError, match='truncated'):
        pipeline.extract_operation_objects('Lift a bottle.')


def test_duplicate_phrases_removed():
    assert split_object_phrases('robot, robot arm, bottle, bottle') == ['robot', 'robot arm', 'bottle']
