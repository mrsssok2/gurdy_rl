# Gnani Speech API notes

Read 2026-11-08 from https://docs.gnani.ai/api/introduction/introduction and these linked pages:
quick-start, STT/speech-to-text, STT/models/prisma-v3, TTS/tts-inference, TTS/available-voices.
We use only the Speech APIs (no Agent Builder / api.inya.ai).

## Common
- Base URL: `https://api.vachana.ai`
- Auth: header `X-API-Key-ID: <key>` on every request. Missing or bad key → 401.
- Auth and rate-limit errors return `{"detail": {"error_code", "message", "status_code"}}`:
  401 `MISSING_API_KEY`/`INVALID_API_KEY`, 402 insufficient credit, 403 `SCOPE_NOT_ALLOWED`,
  429 `RATE_LIMITED` (no Retry-After header; e.g. only one active request per account).

## STT REST: `POST /stt/v3` (multipart/form-data), synchronous
- `audio_file` (required): WAV/MP3/OGG/FLAC/AAC/M4A, **at most 30 s**, mono only (no multi-channel).
  The server converts to 16 kHz mono. Accepts 8 to 44.1 kHz. Clips under 0.1 s give an empty transcript.
  → We send 16 kHz, mono, 16-bit PCM WAV.
- `model`: `prisma-v3.0`. **If omitted, Prisma v2.5 is used.**
- `language_code` (required*): `indic-auto` = auto-detect (v3.0 only), or 1 to 4 comma-separated hints such as `hi-IN,en-IN`.
- `format`: `verbatim` (default) | `transcribe` (ITN, numbers as digits) | `codemix` (v3) | `transliterate` (v3, Latin script output).
- Optional v3 context fields: `keyterms` (up to 200), `topic` (up to 1600 chars), `previous_text`, `speaker_names`.
- Optional header `X-API-Request-ID`, echoed back as `request_id`.
- 200 response: `success`, `request_id`, `transcript`, `model`, `detected_language_code`,
  `processing_time`, `end_to_end_latency`, and `truncated` when the transcript was cut short.
  `detected_language_code` is judged from the transcript's script and is absent for an empty transcript.
  Hindi and Marathi share Devanagari, so short Marathi clips may come back as `hi-IN`.
- Errors: `{"success": false, "error": {"type", "message"}}`, e.g. 400 `MAX_AUDIO_DURATION_EXCEEDED`,
  `UNSUPPORTED_AUDIO_FORMAT`, `AUDIO_CONVERSION_ERROR`, `INVALID_LANGUAGE_CODE`. A missing field gives 422 `{"detail": [...]}`.
- STT languages: bn, en, gu, hi, kn, ml, mr, pa, ta, te (all `-IN`). Assamese is not supported.
- Also exists, not used yet: WebSocket `wss://api.vachana.ai/stt/v3/stream` (16-bit mono PCM, 8 or 16 kHz,
  1024-byte frames, then `{"type":"commit"}`).

## TTS REST: `POST /api/v1/tts/inference` (JSON)
- Headers: `Content-Type: application/json`, `X-API-Key-ID`.
- Body: `text`, `model` (`timbre-v2.5`; `timbre-v3.0` also exists), `voice` (must belong to that model),
  `language` (`hi-IN`, `en-IN`, …, `hi-en` = Hinglish, or `auto`), `speed` 0.85 to 1.15 (default 1.0),
  `audio_config`: `sample_rate` (8000 to 48000, default 48000), `num_channels` 1, `sample_width` 2,
  `encoding` `linear_pcm`, `container` `wav` | `raw` | `mp3` | `ogg` | …
  (The OpenAPI spec marks `audio_config` as required, so always send it.)
- Write numbers and dates out as words in `text`.
- 200 response = **raw binary audio** (`audio/wav` for the wav container), not JSON or base64.
- Errors: flat `{"success", "message", "status_code"}`. 400 = bad parameter, 500 = e.g. text too long.
- Also exists: SSE `/api/v1/tts/sse` (base64 chunks), WebSocket `wss://api.vachana.ai/api/v1/tts`.

## Timbre v2.5 voices (each has one preferred language)
- Hindi `hi-IN`: female Nalini, Bhavna, Yashvi, Urmila, Jwala, Chitra, Ambuja; male Deepak, Roopesh, Vikrant, Hemraj, Jalaj, Omkar.
- Hinglish `hi-en`: Poorvi (F).  English `en-IN`: Kaveri, Trupti, Devika (F); Pranav, Shlok, Girish (M).
- Sending a v3.0 voice to v2.5, or the reverse, returns 400.
- Our default: **Nalini, `hi-IN`**. Try Poorvi/`hi-en` if romanized Hindi text sounds off.
