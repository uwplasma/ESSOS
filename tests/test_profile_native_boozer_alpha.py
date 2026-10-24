"""Avoid false GPU-profile claims from host launches or duplicate op tracks."""
import gzip
import json

import pytest

import jax.profiler

from examples.benchmarks.profile_native_boozer_alpha import summarize_trace, xplane_resources


def write_trace(tmp_path, events):
    path = tmp_path/"test.trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump(dict(traceEvents=events), stream)
    return path


def metadata():
    return [dict(ph="M", name="process_name", pid=0, args=dict(name="/host:CPU")),
            dict(ph="M", name="process_name", pid=1, args=dict(name="/device:GPU:0")),
            dict(ph="M", name="thread_name", pid=1, tid=10, args=dict(name="Stream #10")),
            dict(ph="M", name="thread_name", pid=1, tid=20, args=dict(name="XLA Ops"))]


def test_host_launches_do_not_establish_gpu_capture(tmp_path):
    events = metadata()+[dict(ph="X", pid=0, tid=0, name="cuLaunchKernel", ts=0, dur=100)]
    report = summarize_trace(write_trace(tmp_path, events))
    assert not report["gpu_stream_events_present"]
    assert not report["kernels"]


def test_duplicate_op_tracks_and_transfers_are_not_kernel_time(tmp_path):
    events = metadata()+[
        dict(ph="X", pid=1, tid=10, name="orbit", ts=0, dur=3000),
        dict(ph="X", pid=1, tid=20, name="duplicate op", ts=0, dur=3000),
        dict(ph="X", pid=1, tid=10, name="MemcpyHtoD", ts=3000, dur=500)]
    report = summarize_trace(write_trace(tmp_path, events))
    assert report["summed_kernel_time_ms"] == 3.
    assert len(report["kernels"]) == 1
    assert report["memcpy_or_memset_events"] == 1


def test_overlapping_kernels_keep_union_separate_from_sum(tmp_path):
    events = metadata()+[
        dict(ph="X", pid=1, tid=10, name="long", ts=0, dur=10000),
        dict(ph="X", pid=1, tid=10, name="short", ts=1000, dur=2000)]
    report = summarize_trace(write_trace(tmp_path, events))
    assert report["summed_kernel_time_ms"] == 12.
    assert report["kernel_interval_union_ms"] == 10.
    assert report["first_to_last_kernel_ms"] == 10.
    assert report["time_without_a_kernel_within_span_ms"] == 0.


def test_gaps_and_copy_time_are_not_called_kernel_work(tmp_path):
    events = metadata()+[
        dict(ph="X", pid=1, tid=10, name="first", ts=0, dur=1000),
        dict(ph="X", pid=1, tid=10, name="MemcpyDtoH", ts=1000, dur=500),
        dict(ph="X", pid=1, tid=10, name="last", ts=3000, dur=1000)]
    report = summarize_trace(write_trace(tmp_path, events))
    assert report["kernel_interval_union_ms"] == 2.
    assert report["first_to_last_kernel_ms"] == 4.
    assert report["time_without_a_kernel_within_span_ms"] == 2.
    assert report["memcpy_or_memset_sum_ms"] == .5


def test_resources_preserve_missing_values_and_deduplicate_launch_ids(tmp_path):
    details = "regs:128 static_shared:0 dynamic_shared:16 grid:8192,1,1 block:32,1,1 occ_pct:25"
    events = metadata()+[dict(ph="X", pid=1, tid=10, name="orbit", ts=i*1000, dur=1000,
        args=dict(correlation_id=str(i), kernel_details=details)) for i in range(2)]
    events.append(dict(ph="X", pid=1, tid=10, name="unknown resources", ts=2000, dur=100))
    report = summarize_trace(write_trace(tmp_path, events))
    row = report["kernels"][0]
    assert row["calls"] == 2
    assert row["percent_of_summed_kernel_time"] == pytest.approx(100*2000/2100)
    assert row["launch_resources"] == [dict(registers_per_thread=128,
        static_shared_bytes=0, dynamic_shared_bytes=16, grid_blocks=[8192,1,1],
        block_threads=[32,1,1], theoretical_occupancy_pct=25.)]
    assert report["kernels"][1]["launch_resources"] == [{}]


def test_native_xplane_recovers_details_missing_from_json(tmp_path):
    reader = jax.profiler.ProfileData
    if reader is None:
        pytest.skip("Native reader is not included in this jaxlib build")
    text = '''planes {
      id: 0 name: "/device:GPU:0"
      lines { id: 10 name: "Stream #10"
        events { metadata_id: 1 duration_ps: 1000000
          stats { metadata_id: 1 str_value: "regs:128 static_shared:0 dynamic_shared:0 grid:8192,1,1 block:32,1,1 occ_pct:25" }
          stats { metadata_id: 2 double_value: 25 }
        }
      }
      event_metadata { key: 1 value { id: 1 name: "kernel__1" } }
      stat_metadata { key: 1 value { id: 1 name: "kernel_details" } }
      stat_metadata { key: 2 value { id: 2 name: "theoretical_occupancy_pct" } }
    }'''
    path = tmp_path/"test.xplane.pb"
    path.write_bytes(reader.text_proto_to_serialized_xspace(text))
    report = xplane_resources(path)
    assert report["available"]
    assert len(report["launches"]) == 1
    launch = report["launches"][0]
    assert launch["name"] == "kernel__1"
    assert launch["resources"] == dict(registers_per_thread=128, static_shared_bytes=0,
        dynamic_shared_bytes=0, grid_blocks=[8192,1,1], block_threads=[32,1,1],
        theoretical_occupancy_pct=25.)


def test_absent_native_reader_is_reported_without_fabricated_resources(monkeypatch, tmp_path):
    monkeypatch.setattr(jax.profiler, "ProfileData", None)
    report = xplane_resources(tmp_path/"absent.xplane.pb")
    assert not report["available"]
    assert not report["launches"]
