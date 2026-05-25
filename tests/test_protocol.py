import socket

from mlx_speechd.protocol import decode_line, encode_message, iter_messages


def test_protocol_roundtrip() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(encode_message({"op": "say", "text": "hello\nworld"}))
        left.shutdown(socket.SHUT_WR)
        assert list(iter_messages(right)) == [{"op": "say", "text": "hello\nworld"}]
    finally:
        left.close()
        right.close()


def test_decode_rejects_non_object() -> None:
    try:
        decode_line(b"[]")
    except Exception as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("expected decode failure")
