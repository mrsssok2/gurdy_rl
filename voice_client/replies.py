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
# Backward and sideways walks (scripted only), in romanized Hindi; their
# replies are first person
DIRECTION_NAMES = {"walk_backward": "peeche", "walk_left": "baayen", "walk_right": "daayen"}
WALK_ACTIONS = ("walk_forward",) + tuple(DIRECTION_NAMES)

# Camera replies. Gend (ball) and cheez (thing) are feminine, dabba (box)
# masculine; laal doesn't change, neela/neeli and hara/hari do. Before a
# postposition (ke paas, ki taraf, tak) the masculine forms are oblique:
# "neele dabbe ke paas". Keys: gender, or "o" for masculine oblique.
OBJECT_NAMES = {"ball": ("gend", "f"), "box": ("dabba", "m"), None: ("cheez", "f")}
OBLIQUE_NOUNS = {"dabba": "dabbe"}
COLOR_NAMES = {"red": {"m": "laal", "f": "laal", "o": "laal"},
               "blue": {"m": "neela", "f": "neeli", "o": "neele"},
               "green": {"m": "hara", "f": "hari", "o": "hare"}}
SIDE_NAMES = {"center": "saamne", "left": "baayen taraf", "right": "daayen taraf"}
SIDE_ORDER = ("center", "left", "right")
NO_CAMERA_REPLY = "Mujhe camera se tasveer nahi mil rahi. Kya vision node chal raha hai?"


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


def _first_person_reply(status, dist, side, feedback, backwards):
    """
    First-person reply for a backward or sideways walk (scripted only, no
    fallback). side is the direction word, backwards True if the robot went
    the opposite way.
    """
    if status == "done":
        return "Main %s %s chala." % (dist, side)
    if status == "stopped":
        return "Main ruk gaya. Main %s %s chala." % (dist, side)
    if status == "fallen":
        text = "Main %s %s chalne ke baad gir gaya." % (dist, side)
        if feedback.get("reset"):
            text += " Mujhe wapas khada kar diya gaya."
        return text
    if status == "stuck":
        return "Main %s nahi badh paaya, sirf %s %s." % (
            side, dist, "ulti taraf gaya" if backwards else "chala")
    if status == "step_limit":
        return "Main bahut dheere chala. Main %s hi %s chal paaya." % (dist, side)
    return None


def _walk_reply(feedback, parsed):
    status = feedback.get("status")
    walked = feedback.get("distance_m", 0.0)
    dist = distance_words(walked)
    parts = []
    if parsed.get("capped"):
        parts.append("Gurdy ek baar mein pachaas centimeter tak hi chalta hai.")
    backwards = walked < -0.005

    side = DIRECTION_NAMES.get(feedback.get("action"))
    if side and status != "busy":
        text = _first_person_reply(status, dist, side, feedback, backwards)
        if text is None:
            return "Robot ne error diya."
        parts.append(text)
        return " ".join(parts)

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


def object_phrase(color, shape, oblique=False):
    """
    ("neela dabba", "m") for ("blue", "box"), ("neele dabbe", "m") with
    oblique (before ke paas / ki taraf / tak); shape None is "cheez".
    """
    noun, gender = OBJECT_NAMES.get(shape, OBJECT_NAMES[None])
    form = gender
    if oblique and gender == "m":
        noun, form = OBLIQUE_NOUNS.get(noun, noun), "o"
    adjective = COLOR_NAMES.get(color, {}).get(form, color or "")
    return ("%s %s" % (adjective, noun)).strip(), gender


def _dikh(gender, negative=False):
    verb = "rahi" if gender == "f" else "raha"
    return ("nahi dikh %s" if negative else "dikh %s hai") % verb


def join_words(parts):
    """'a', 'a aur b', 'a, b aur c'."""
    if len(parts) <= 1:
        return "".join(parts)
    return "%s aur %s" % (", ".join(parts[:-1]), parts[-1])


def _describe_reply(feedback):
    if feedback.get("status") == "no_camera":
        return NO_CAMERA_REPLY
    # One mention per colour and shape (the largest, which comes first)
    seen, items = set(), []
    for obj in feedback.get("objects") or ():
        key = (obj.get("color"), obj.get("shape"))
        if key in seen or obj.get("color") not in COLOR_NAMES:
            continue
        seen.add(key)
        items.append(obj)
    if not items:
        return "Mujhe abhi saamne koi laal, neeli ya hari cheez nahi dikh rahi."
    items.sort(key=lambda o: SIDE_ORDER.index(o.get("side"))
               if o.get("side") in SIDE_ORDER else len(SIDE_ORDER))
    parts, gender = [], "m"
    for obj in items:
        phrase, gender = object_phrase(obj.get("color"), obj.get("shape"))
        side = SIDE_NAMES.get(obj.get("side"), "")
        parts.append(("%s %s" % (side, phrase)).strip())
    # The verb agrees with the nearest (last) object, as spoken Hindi does
    return "Mujhe %s %s." % (join_words(parts), _dikh(gender))


def _go_to_reply(feedback):
    status = feedback.get("status")
    phrase, gender = object_phrase(feedback.get("color"), feedback.get("shape"))
    # Before ke paas / ki taraf / tak
    obl, _ = object_phrase(feedback.get("color"), feedback.get("shape"), oblique=True)
    gaya = "gayi" if gender == "f" else "gaya"
    if status == "reached":
        return "Main %s ke paas pahunch gaya." % obl
    if status == "not_found":
        return "Mujhe %s %s." % (phrase, _dikh(gender, negative=True))
    if status == "lost":
        return "%s nazar se kho %s, isliye main ruk gaya." % (phrase[0].upper() + phrase[1:], gaya)
    if status == "stopped":
        return "Main ruk gaya. Main %s ki taraf %s chala." % (
            obl, distance_words(feedback.get("distance_m", 0.0)))
    if status == "fallen":
        text = "Main %s ki taraf jaate hue gir gaya." % obl
        if feedback.get("reset"):
            text += " Mujhe wapas khada kar diya gaya."
        return text
    if status == "stuck":
        return "Main %s ki taraf aage nahi badh paaya." % obl
    if status == "gave_up":
        return "Main %s tak nahi pahunch paaya." % obl
    if status == "no_camera":
        return NO_CAMERA_REPLY
    if status == "busy":
        return "Gurdy abhi chal raha hai. Pehle use rukne dijiye."
    return "Robot ne error diya."


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
    elif intent == "unknown" and parsed.get("unsupported"):
        body = ("Gurdy sirf aage, peeche, baayen ya daayen chal sakta hai. "
                "Boliye, Gurdy baayen bees centimeter chalo.")
    elif intent == "unknown":
        body = "Yeh command samajh nahi aaya. Boliye, Gurdy aage bees centimeter chalo."
    elif feedback is None:
        body = "Robot se jawab nahi aaya."
    elif feedback.get("status") == "error":
        body = "Robot ne error diya."
    elif intent == "status":
        body = _status_reply(feedback)
    elif intent == "describe":
        body = _describe_reply(feedback)
    elif intent == "go_to_object" or (intent == "stop" and
                                      feedback.get("action") == "go_to_object"):
        body = _go_to_reply(feedback)
    elif intent == "stop":
        if feedback.get("action") in WALK_ACTIONS:
            body = _walk_reply(feedback, parsed)
        else:
            body = "Gurdy pehle se ruka hua hai."
    elif intent in WALK_ACTIONS:
        body = _walk_reply(feedback, parsed)
    else:
        body = "Yeh command samajh nahi aaya."
    return "%s %s" % (PREFIX, body)
