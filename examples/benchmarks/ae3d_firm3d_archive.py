"""Retrieve a pinned, small AE3D/FIRM3D reference packet, without executing it.

Read a local copy of the published ZIP, or explicitly request HTTP ranges.
Every member is checked against its pinned name, sizes, method and ZIP CRC;
the complete archive's MD5 is provenance only when fetching ranges.
"""

import argparse
import binascii
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile


MANIFEST = Path(__file__).with_name("references")/"ae3d_firm3d_qa.json"


def decode_member(record, payload):
    """Validate and decode one bounded ZIP local record using the stdlib."""
    if len(payload) < 30 or payload[:4] != b"PK\x03\x04":
        raise ValueError("Missing ZIP local header")
    header = struct.unpack_from("<4s5H3I2H", payload)
    _, version, flags, method, _, _, crc, compressed, size, name_len, extra_len = header
    if flags & (1 | 8) or version >= 45:
        raise ValueError("Encrypted, streaming-descriptor and ZIP64 members are unsupported")
    name = payload[30:30+name_len].decode("utf-8")
    if (name != record["name"] or method != record["method"] or crc != record["crc"]
            or compressed != record["compressed_size"] or size != record["size"]):
        raise ValueError("ZIP member differs from pinned metadata")
    end = 30+name_len+extra_len+compressed
    if end > len(payload):
        raise ValueError("Truncated ZIP member")
    # Give ZipFile just this local record, with a matching synthetic directory.
    central = struct.pack("<4s6H3I5H2I", b"PK\x01\x02", 20, version, flags, method,
        0, 0, crc, compressed, size, name_len, 0, 0, 0, 0, 0, 0)+payload[30:30+name_len]
    eocd = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, len(central), end, 0)
    with zipfile.ZipFile(io.BytesIO(payload[:end]+central+eocd)) as archive:
        data = archive.read(name)  # ZIP CRC is independently checked by ZipFile.
    if len(data) != size or binascii.crc32(data) != crc:
        raise ValueError("Member size/CRC mismatch")
    return data


def fetch_packet(output, *, archive_path=None, manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text())
    members = manifest["members"]
    if sum(item["size"] for item in members) > 20_000_000:
        raise ValueError("Reference packet exceeds the 20 MB decompressed budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes, network_bytes = {}, 0
    if archive_path is not None:
        if Path(archive_path).stat().st_size != manifest["archive_size"]:
            raise ValueError("Local archive size differs from the pinned release")
        local = zipfile.ZipFile(archive_path)
    else:
        local = None
    try:
        for record in members:
            relative = Path(record["name"])
            destination = (output/relative).resolve()
            if not destination.is_relative_to(output.resolve()):
                raise ValueError("Unsafe archive path")
            if destination.exists():
                data = destination.read_bytes()
                if len(data) != record["size"] or binascii.crc32(data) != record["crc"]:
                    raise ValueError(f"Existing file differs from reference: {relative}")
            elif local is not None:
                info = local.getinfo(record["name"])
                if info.CRC != record["crc"] or info.file_size != record["size"]:
                    raise ValueError("Local member differs from pinned release")
                data = local.read(info)
            else:
                start = record["offset"]
                # Local filename and extra field follow the 30-byte header.
                length = 30+len(record["name"].encode())+4096+record["compressed_size"]
                end = min(start+length-1, manifest["archive_size"]-1)
                request = urllib.request.Request(manifest["archive_url"],
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "ESSOS-reference-packet"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected_range = f"bytes {start}-{end}/{manifest['archive_size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Server did not honor the bounded range; refusing a full archive download")
                    payload = response.read(length+1)
                if len(payload) != end-start+1:
                    raise ValueError("HTTP range has an unexpected length")
                network_bytes += len(payload)
                data = decode_member(record, payload)
            expected_sha = record.get("sha256")
            digest = hashlib.sha256(data).hexdigest()
            if expected_sha is not None and digest != expected_sha:
                raise ValueError("SHA256 differs from the pinned packet")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
            hashes[record["name"]] = digest
            print(f"Verified {relative.name}: {len(data)} bytes", flush=True)
    finally:
        if local is not None:
            local.close()
    report = dict(doi=manifest["doi"], network_bytes=network_bytes, sha256=hashes,
                  verification="Per-member pinned ZIP CRC and any supplied SHA256; whole-ZIP MD5 not computed")
    (output/"packet_hashes.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3]/"artifacts/fax_integration/ae3d_firm3d_packet")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Use a local published ZIP, with no network access")
    source.add_argument("--download", action="store_true", help="Download only the pinned members using HTTP ranges")
    args = parser.parse_args()
    fetch_packet(args.output, archive_path=args.archive)


if __name__ == "__main__":
    main()
