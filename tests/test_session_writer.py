# tests/test_session_writer.py
#
# Unit tests for SessionWriter — no hardware required.

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

import pytest

# Patch SESSIONS_DIR to a temp directory for all tests in this module
import config


class TestSessionWriter:

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Redirect session output to temp directory
        self._original_sessions_dir = config.SESSIONS_DIR
        config.SESSIONS_DIR = self.tmp_dir

    def teardown_method(self):
        config.SESSIONS_DIR = self._original_sessions_dir

    def _make_writer(self):
        # Import here so the patched SESSIONS_DIR is active
        from src.output.session import SessionWriter
        return SessionWriter(
            loopback_device_name="Test Loopback Device",
            mic_device_name="Test Microphone Device",
        )

    def test_session_file_created_on_init(self):
        writer = self._make_writer()
        assert writer.filepath.exists()

    def test_schema_version_correct(self):
        writer = self._make_writer()
        with open(writer.filepath) as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"

    def test_devices_recorded(self):
        writer = self._make_writer()
        with open(writer.filepath) as f:
            data = json.load(f)
        assert data["devices"]["loopback"] == "Test Loopback Device"
        assert data["devices"]["microphone"] == "Test Microphone Device"

    def test_ended_at_null_before_close(self):
        writer = self._make_writer()
        with open(writer.filepath) as f:
            data = json.load(f)
        assert data["ended_at"] is None

    def test_ended_at_populated_after_close(self):
        writer = self._make_writer()
        writer.close()
        with open(writer.filepath) as f:
            data = json.load(f)
        assert data["ended_at"] is not None

    def test_append_segment_increments_id(self):
        writer = self._make_writer()
        id1 = writer.append_segment("Guest", "Hello.", 0.95, "loopback")
        id2 = writer.append_segment("Host", "Hi there.", 0.92, "microphone")
        assert id1 == 1
        assert id2 == 2

    def test_append_segment_persisted_to_file(self):
        writer = self._make_writer()
        writer.append_segment("Guest", "Tell me about yourself.", 0.97, "loopback")
        with open(writer.filepath) as f:
            data = json.load(f)
        assert len(data["segments"]) == 1
        seg = data["segments"][0]
        assert seg["text"] == "Tell me about yourself."
        assert seg["speaker"] == "Guest"
        assert seg["confidence"] == 0.97
        assert seg["source_stream"] == "loopback"

    def test_empty_segments_on_init(self):
        writer = self._make_writer()
        with open(writer.filepath) as f:
            data = json.load(f)
        assert data["segments"] == []
