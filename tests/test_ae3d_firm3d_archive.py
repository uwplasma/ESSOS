"""Bounded archive reads preserve pinned scientific inputs and never run them."""

import io
import json
import zipfile

import pytest

from examples.benchmarks.ae3d_firm3d_archive import decode_member, fetch_packet


def fixture(tmp_path, name="packet/input.txt"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, b"independent expected reference data\n")
        info = archive.getinfo(name)
    record = dict(name=name, size=info.file_size, compressed_size=info.compress_size,
                  method=info.compress_type, crc=info.CRC, offset=info.header_offset)
    payload = stream.getvalue()
    local = tmp_path/"reference.zip"
    local.write_bytes(payload)
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(doi="test fixture", archive_size=len(payload),
        archive_url="https://invalid.example/no-network-in-tests", members=[record])))
    return record, payload, local, manifest


def test_local_and_individual_member_reads_recover_the_same_verified_bytes(tmp_path):
    record, payload, local, manifest = fixture(tmp_path)
    assert decode_member(record, payload) == b"independent expected reference data\n"
    output = tmp_path/"output"
    report = fetch_packet(output, archive_path=local, manifest_path=manifest)
    assert report["network_bytes"] == 0
    assert len(report["sha256"][record["name"]]) == 64
    # A repeated invocation verifies existing data instead of overwriting it.
    assert fetch_packet(output, archive_path=local, manifest_path=manifest) == report
    (output/record["name"]).write_text("changed data")
    with pytest.raises(ValueError, match="Existing file"):
        fetch_packet(output, archive_path=local, manifest_path=manifest)


@pytest.mark.parametrize("change", ["name", "crc", "truncated"])
def test_changed_or_truncated_records_are_rejected(tmp_path, change):
    record, payload, _, _ = fixture(tmp_path)
    if change == "name": record["name"] = "different/input.txt"
    if change == "crc": record["crc"] ^= 1
    if change == "truncated": payload = payload[:31]
    with pytest.raises(ValueError):
        decode_member(record, payload)


def test_member_paths_cannot_leave_the_packet_directory(tmp_path):
    _, _, local, manifest = fixture(tmp_path, "../outside.txt")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        fetch_packet(tmp_path/"output", archive_path=local, manifest_path=manifest)
    assert not (tmp_path/"outside.txt").exists()


def test_server_ignoring_range_is_rejected_before_reading_body(tmp_path, monkeypatch):
    _, _, _, manifest = fixture(tmp_path)

    class FullDownload:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, *args): pytest.fail("A full archive body must never be read")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FullDownload())
    with pytest.raises(ValueError, match="refusing a full archive"):
        fetch_packet(tmp_path/"output", manifest_path=manifest)
