"""Keyword parser for Gurdy voice commands in Devanagari Hindi, romanized
Hindi/Hinglish and English.

parse(transcript) returns a dict with at least
    intent     "walk_forward" | "stop" | "status" | "mode" | "reset" | "unknown"
    wake_word  True if the transcript started with (or contained) "Gurdy"
    text       the transcript as given
plus, per intent:
    walk_forward  distance_m, distance_cm, defaulted (no distance heard),
                  capped (asked for more than MAX_CM)
    mode          mode: "rl" | "scripted"

Example: "गरदी आगे बीस सेंटीमीटर चलो।" -> walk_forward, 0.2 m.

The wake word is optional. Intents are checked in the order reset, stop,
status, mode, walk, so "kitna aage chala?" is a status question and "ruko, aage mat
chalo" is a stop.
"""

import re

DEFAULT_CM = 20
MAX_CM = 50

# Wake word spellings seen from (or expected of) the STT
WAKE_WORDS = {
    "गरदी", "गर्डी", "गुर्डी", "गुरदी", "गर्दी", "गुर्दी", "गरडी", "गुरडी", "गर्डि", "गार्डी",
    "gurdy", "gurdi", "gurady", "gardi", "gurdie", "gurddy", "girdy", "gerdy",
}

STOP_WORDS = {
    "रुको", "रुक", "रूको", "रुकें", "रुकिए", "रोको", "रुकजाओ", "थमो", "ठहरो",
    "ruko", "ruk", "rukko", "roko", "rukjao", "stop", "halt", "स्टॉप", "स्टाप",
}

# Client-side: clear a stuck "busy" state ("wapas" / "reset")
RESET_WORDS = {"reset", "wapas", "vapas", "wapis", "vapis", "वापस", "वापिस", "रीसेट"}

STATUS_WORDS = {"status", "स्टेटस", "स्टैटस"}
# Two-word status phrases, matched as consecutive tokens
STATUS_PHRASES = {
    ("कितना", "चला"), ("कितना", "चले"), ("कितना", "चल"), ("कितनी", "दूर"),
    ("kitna", "chala"), ("kitna", "chale"), ("kitna", "chal"), ("kitni", "door"),
    ("kitni", "dur"), ("how", "far"),
}
# "kitna" + anything up to two tokens later + chala, e.g. "kitna aage chala"
STATUS_QUESTION = {"कितना", "कितनी", "kitna", "kitni", "kitana"}
STATUS_VERBS = {"चला", "चले", "चल", "chala", "chale", "chal", "door", "dur", "दूर"}

MODE_WORDS = {"mode", "मोड", "मोड़", "मोड़", "मौड"}
RL_WORDS = {"rl", "आरएल", "आर.एल", "rlmode"}
# "आर एल" / "r l" as two tokens
RL_PAIRS = {("आर", "एल"), ("r", "l"), ("are", "l"), ("aar", "el"), ("r", "el")}
NORMAL_WORDS = {"normal", "नॉर्मल", "नार्मल", "नोर्मल", "नॉरमल", "scripted", "स्क्रिप्टेड"}

FORWARD_WORDS = {
    "आगे", "आगें", "aage", "aagey", "aghe", "aaghe", "age", "agey", "aange",
    "forward", "फॉरवर्ड", "फॉर्वर्ड", "ahead",
}
WALK_VERBS = {
    "चलो", "चल", "चलिए", "चलें", "बढ़ो", "बढ़ो", "बढ़", "जाओ",
    "chalo", "chal", "chaliye", "badho", "barho", "jao", "walk", "move", "go",
}

CM_UNITS = {
    "सेंटीमीटर", "सेंटिमीटर", "सेन्टीमीटर", "सेंटीमिटर", "सेमी", "सेंमी", "सेंटी",
    "cm", "cms", "centimeter", "centimeters", "centimetre", "centimetres", "senti",
}
M_UNITS = {"मीटर", "मिटर", "meter", "meters", "metre", "metres", "m", "mtr"}

# Number words. Distances are usually tens of cm; the small numbers are for
# metres ("ek meter").
NUMBER_WORDS = {
    # Devanagari
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "पंद्रह": 15, "पन्द्रह": 15,
    "बीस": 20, "पच्चीस": 25, "तीस": 30, "पैंतीस": 35, "चालीस": 40,
    "पैंतालीस": 45, "पचास": 50, "सौ": 100, "आधा": 0.5,
    # Romanized Hindi
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4, "paanch": 5,
    "panch": 5, "paach": 5, "chhah": 6, "che": 6, "saat": 7, "sat": 7, "aath": 8,
    "nau": 9, "das": 10, "dus": 10, "pandrah": 15, "pandra": 15, "bees": 20,
    "bis": 20, "beas": 20, "pachchis": 25, "pachis": 25, "pachees": 25,
    "tees": 30, "tis": 30, "paintees": 35, "chalis": 40, "chaalis": 40,
    "chalees": 40, "paintalis": 45, "pachaas": 50, "pachas": 50, "sau": 100,
    "aadha": 0.5, "adha": 0.5,
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "half": 0.5,
}
# "twenty five", "bees paanch" is not Hindi, so only English compounds
ENGLISH_TENS = {"twenty": 20, "thirty": 30, "forty": 40}
ENGLISH_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9}

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Punctuation to drop, including the danda (।) and double danda (॥)
_PUNCT = re.compile(r"[।॥,!?;:\"'()\[\]{}…\-–—]+")
_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def tokenize(text):
    """
    Lower-case, ASCII-digit tokens. Splits glued digits and units ("20cm"),
    drops punctuation and dots that end a word, and keeps Devanagari words
    with their vowel signs whole.
    """
    text = text.translate(DEVANAGARI_DIGITS).lower()
    text = _PUNCT.sub(" ", text)
    # "20cm" -> "20 cm", "cm20" -> "cm 20"; keep "0.5" whole
    text = re.sub(r"(\d)(?=[^\d\s.])", r"\1 ", text)
    text = re.sub(r"([^\d\s.])(?=\d)", r"\1 ", text)
    tokens = []
    for token in text.split():
        if not _NUMBER.match(token):
            # "r.l." -> "rl", "cm." -> "cm"
            token = token.replace(".", "")
        if token:
            tokens.append(token)
    return tokens


def _number_at(tokens, i):
    """(value, tokens used) of a number at tokens[i], or (None, 0)."""
    token = tokens[i]
    if _NUMBER.match(token):
        return float(token), 1
    if token in ENGLISH_TENS and i + 1 < len(tokens) and tokens[i + 1] in ENGLISH_ONES:
        return float(ENGLISH_TENS[token] + ENGLISH_ONES[tokens[i + 1]]), 2
    if token in NUMBER_WORDS:
        return float(NUMBER_WORDS[token]), 1
    return None, 0


def find_distance_cm(tokens):
    """
    Distance in cm from the first number in tokens, or None. A unit right
    after the number, or anywhere if there is none there, picks cm or
    metres; with no unit at all the number is cm.
    """
    for i in range(len(tokens)):
        value, used = _number_at(tokens, i)
        if value is None:
            continue
        # "do" is also the Hindi verb "give" and "go" in Hinglish ("aage
        # chalo do"); only count it as a number when a unit follows
        nxt = tokens[i + used] if i + used < len(tokens) else None
        if tokens[i] in ("do", "go") and nxt not in CM_UNITS | M_UNITS:
            continue
        if nxt in M_UNITS:
            return value * 100.0
        if nxt in CM_UNITS:
            return value
        if any(t in CM_UNITS for t in tokens):
            return value
        if any(t in M_UNITS for t in tokens):
            return value * 100.0
        return value
    return None


def _has_pair(tokens, pairs):
    return any((a, b) in pairs for a, b in zip(tokens, tokens[1:]))


def _is_status(tokens):
    if any(t in STATUS_WORDS for t in tokens) or _has_pair(tokens, STATUS_PHRASES):
        return True
    for i, token in enumerate(tokens):
        if token in STATUS_QUESTION and any(t in STATUS_VERBS for t in tokens[i + 1:i + 4]):
            return True
    return False


def _mode(tokens):
    """ "rl" or "scripted" for a mode switch, else None."""
    if not any(t in MODE_WORDS for t in tokens):
        return None
    if any(t in RL_WORDS for t in tokens) or _has_pair(tokens, RL_PAIRS):
        return "rl"
    if any(t in NORMAL_WORDS for t in tokens):
        return "scripted"
    return None


def parse(transcript):
    """Parse one STT transcript into a command dict (see the module doc)."""
    text = transcript or ""
    tokens = tokenize(text)
    wake = any(t in WAKE_WORDS for t in tokens)
    tokens = [t for t in tokens if t not in WAKE_WORDS]
    result = {"text": text, "wake_word": wake}

    if any(t in RESET_WORDS for t in tokens):
        result["intent"] = "reset"
        return result
    if any(t in STOP_WORDS for t in tokens):
        result["intent"] = "stop"
        return result
    if _is_status(tokens):
        result["intent"] = "status"
        return result
    mode = _mode(tokens)
    if mode:
        result.update(intent="mode", mode=mode)
        return result

    forward = any(t in FORWARD_WORDS for t in tokens)
    walk_verb = any(t in WALK_VERBS for t in tokens)
    cm = find_distance_cm(tokens)
    # "aage chalo", "aage bees cm", or "bees cm chalo"
    if forward or (walk_verb and cm is not None):
        defaulted = cm is None or cm <= 0
        if defaulted:
            cm = DEFAULT_CM
        capped = cm > MAX_CM
        cm = min(cm, MAX_CM)
        result.update(intent="walk_forward", distance_cm=cm,
                      distance_m=round(cm / 100.0, 4),
                      defaulted=defaulted, capped=capped)
        return result

    result["intent"] = "unknown"
    return result


if __name__ == "__main__":
    import json
    import sys
    for line in sys.argv[1:] or sys.stdin:
        print(json.dumps(parse(line.strip()), ensure_ascii=False))
