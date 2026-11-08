"""Short Hinglish replies for the TTS (Timbre, voice Poorvi, language hi-en).

Every reply starts with PREFIX. Numbers are written out in romanized Hindi,
as the TTS docs ask for words rather than digits.
"""

PREFIX = "Main ek AI assistant hoon."

# 0..100 in romanized Hindi
HINDI_NUMBERS = (
    "shunya ek do teen chaar paanch chhah saat aath nau das "
    "gyaarah baarah terah chaudah pandrah solah satrah athaarah unnees bees "
    "ikkees baaees teyees chaubees pachchees chhabbees sattaaees atthaaees untees tees "
    "ikattees battees taintees chauntees paintees chhattees saintees adtees untaalees chaalees "
    "iktaalees bayaalees taintaalees chavaalees paintaalees chhiyaalees saintaalees adtaalees unchaas pachaas "
    "ikyaavan baavan tirpan chauvan pachpan chhappan sattaavan atthaavan unsath saath "
    "iksath baasath tirsath chaunsath painsath chhiyaasath sadsath adsath unhattar sattar "
    "ikhattar bahattar tihattar chauhattar pachhattar chhihattar sathattar athhattar unyaasi assi "
    "ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sattaasi athaasi navaasi nabbe "
    "ikyaanve baanve tiraanve chauraanve pachaanve chhiyaanve sattaanve atthaanve ninyaanve sau"
).split()
assert len(HINDI_NUMBERS) == 101

MODE_NAMES = {"rl": "RL mode", "scripted": "normal mode"}


def number_words(n):
    """Romanized Hindi for an integer 0..100; digits beyond that."""
    n = int(n)
    if 0 <= n <= 100:
        return HINDI_NUMBERS[n]
    return str(n)


def distance_words(metres):
    """'bees centimeter', 'ek point paanch meter', ... for a distance in m."""
    metres = abs(float(metres or 0.0))
    cm = int(round(metres * 100))
    if cm <= 100:
        return "%s centimeter" % number_words(cm)
    whole, tenth = divmod(int(round(metres * 10)), 10)
    if tenth:
        return "%s point %s meter" % (number_words(whole), number_words(tenth))
    return "%s meter" % number_words(whole)


def _walk_reply(feedback, parsed):
    status = feedback.get("status")
    walked = feedback.get("distance_m", 0.0)
    dist = distance_words(walked)
    parts = []
    if parsed.get("capped"):
        parts.append("Gurdy ek baar mein pachaas centimeter tak hi chalta hai.")
    backwards = walked < -0.005

    if status == "done":
        mode = MODE_NAMES.get(feedback.get("mode_used"), "normal mode")
        if feedback.get("mode_used") == "rl+scripted":
            parts.append("Gurdy %s aage chala." % dist)
        else:
            parts.append("Gurdy %s aage chala, %s mein." % (dist, mode))
    elif status == "stopped":
        parts.append("Gurdy ruk gaya. Woh %s chala." % dist)
    elif status == "fallen":
        parts.append("Gurdy %s chalne ke baad gir gaya." % dist)
        if feedback.get("reset"):
            parts.append("Maine use wapas khada kar diya.")
    elif status == "stuck":
        parts.append("Gurdy aage nahi badh paaya, sirf %s %s." % (
            dist, "peeche gaya" if backwards else "chala"))
    elif status == "step_limit":
        parts.append("Gurdy bahut dheere chala. Woh %s hi chal paaya." % dist)
    elif status == "busy":
        parts.append("Gurdy abhi chal raha hai. Pehle use rukne dijiye.")
    else:
        return "Robot ne error diya."

    if feedback.get("fallback") and status not in ("busy",):
        if feedback.get("mode_used") == "rl+scripted":
            parts.append("RL mode se progress nahi hui, isliye maine normal mode par switch kiya.")
        else:
            parts.append("RL model nahi mila, isliye maine normal mode use kiya.")
    return " ".join(parts)


def _status_reply(feedback):
    total = distance_words(feedback.get("total_distance_m", 0.0))
    if feedback.get("status") == "walking":
        return "Gurdy abhi chal raha hai, is baar %s ho gaya. Kul %s." % (
            distance_words(feedback.get("distance_m", 0.0)), total)
    return "Gurdy ab tak kul %s chala hai. Abhi woh ruka hua hai." % total


def build_reply(parsed, feedback=None, mode="scripted"):
    """
    Reply text for a parsed command (command_parser.parse) and the robot's
    feedback dict (None if there was no reply). mode is the client's current
    walk mode, used for mode switches.
    """
    intent = parsed.get("intent")
    if intent == "mode":
        body = "Ab %s on hai." % MODE_NAMES[parsed["mode"]]
    elif intent == "reset":
        body = "Theek hai, reset ho gaya. Naya command boliye."
    elif intent == "unknown":
        body = "Yeh command samajh nahi aaya. Boliye, Gurdy aage bees centimeter chalo."
    elif feedback is None:
        body = "Robot se jawab nahi aaya."
    elif feedback.get("status") == "error":
        body = "Robot ne error diya."
    elif intent == "status":
        body = _status_reply(feedback)
    elif intent == "stop":
        if feedback.get("action") == "walk_forward":
            body = _walk_reply(feedback, parsed)
        else:
            body = "Gurdy pehle se ruka hua hai."
    elif intent == "walk_forward":
        body = _walk_reply(feedback, parsed)
    else:
        body = "Yeh command samajh nahi aaya."
    return "%s %s" % (PREFIX, body)
