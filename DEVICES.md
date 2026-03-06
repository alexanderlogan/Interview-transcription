# Device Configuration

Interview Transcriber captures audio via a single WASAPI loopback stream.
Device selection is name-based — the application searches for a device whose
name contains the configured substring.

## Current Configuration

Configured in `config.py` (not shown — set in your local environment):

| Setting | Value | Matches |
|---------|-------|---------|
| `LOOPBACK_DEVICE_NAME` | `SK200` | Speakers (USB SK200) [Loopback] |

## Changing Devices

If you use different audio hardware, update `config.py`:

```python
LOOPBACK_DEVICE_NAME = "YOUR_DEVICE_NAME"
```

Use a substring that uniquely identifies your output device. To find the
correct name, run the diagnostic script:

```powershell
python diag_loopback_callback.py
```

This will list all available audio devices with their full names and indices.

## Common Device Name Substrings

| Hardware | Substring to use |
|----------|-----------------|
| Speakers (USB SK200) | `SK200` |
| Headphones (Generic USB) | part of the device name |
| Realtek Audio | `Realtek` |
| NVIDIA HDMI | `NVIDIA` |

## Notes

- Only loopback devices (marked `[Loopback]`) can capture system audio
- The device must be your active default output device
- Sample rate is read automatically from the device at runtime
- If multiple devices match the substring, the first match is used
