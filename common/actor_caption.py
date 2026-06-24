"""Actor-aware caption prompts and post-processing for s8."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

ActorRef = Union[str, Dict[str, Any]]


def _slugify(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_actor_gender_map(
    gender_map: Dict[str, str] | None,
) -> Dict[str, str]:
    if not gender_map:
        return {}
    out: Dict[str, str] = {}
    for key, gender in gender_map.items():
        slug = _slugify(key)
        g = str(gender).strip().lower()
        if slug and g in {"male", "female", "m", "f"}:
            out[slug] = "male" if g in {"male", "m"} else "female"
    return out


def actor_gender_map_from_config(config: Dict[str, Any] | None) -> Dict[str, str]:
    if not config:
        return {}
    mp = (
        config.get("pipeline", {}).get("master_pipeline")
        or config.get("master_pipeline")
        or {}
    )
    return normalize_actor_gender_map(mp.get("actor_gender_map"))


def _gender_for_actor_name(
    name: str,
    rec: Dict[str, Any],
    gender_map: Dict[str, str] | None = None,
) -> Optional[str]:
    gmap = normalize_actor_gender_map(gender_map)
    g = _gender_for_display_name(name, gmap)
    if g:
        return g
    slug = _slugify(name)
    for actor in rec.get("actors") or []:
        if not isinstance(actor, dict):
            continue
        actor_slug = _slugify(str(actor.get("actor", "")))
        display = str(actor.get("display_name", "")).strip()
        if actor_slug != slug and _slugify(display) != slug:
            continue
        face_gender = str(actor.get("face_gender", "")).strip().lower()
        if face_gender in {"male", "female"}:
            return face_gender
    return None


def _gender_for_display_name(name: str, gender_map: Dict[str, str]) -> Optional[str]:
    return gender_map.get(_slugify(name))


def _man_replacement_patterns(name: str) -> List[tuple[str, str]]:
    return [
        (r"\bthe man's\b", f"{name}'s"),
        (r"\bThe man's\b", f"{name}'s"),
        (r"\bthe man\b", name),
        (r"\bThe man\b", name),
        (r"\ba man\b", name),
        (r"\bA man\b", name),
        (r"\banother man\b", name),
        (r"\bAnother man\b", name),
    ]


def _woman_replacement_patterns(name: str) -> List[tuple[str, str]]:
    return [
        (r"\bthe woman's\b", f"{name}'s"),
        (r"\bThe woman's\b", f"{name}'s"),
        (r"\bthe woman\b", name),
        (r"\bThe woman\b", name),
        (r"\ba woman\b", name),
        (r"\bA woman\b", name),
        (r"\banother woman\b", name),
        (r"\bAnother woman\b", name),
    ]


def _person_replacement_patterns(name: str) -> List[tuple[str, str]]:
    return [
        (r"\bthe person\b", name),
        (r"\bThe person\b", name),
        (r"\ba person\b", name),
        (r"\bA person\b", name),
    ]


def _other_person_phrase(gender: Optional[str]) -> str:
    if gender == "male":
        return "another man"
    if gender == "female":
        return "another woman"
    return "another person"


def _collect_person_patterns(name: str, gender: Optional[str]) -> List[tuple[str, str]]:
    if gender == "male":
        return _man_replacement_patterns(name) + _person_replacement_patterns(name)
    if gender == "female":
        return _woman_replacement_patterns(name) + _person_replacement_patterns(name)
    return (
        _woman_replacement_patterns(name)
        + _man_replacement_patterns(name)
        + _person_replacement_patterns(name)
    )


def _replace_first_person_reference(
    text: str,
    name: str,
    gender: Optional[str],
) -> str:
    """Replace only the first generic person phrase with the identified actor."""
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_repl = ""
    for pattern, repl in _collect_person_patterns(name, gender):
        m = re.search(pattern, text)
        if m and (best_start is None or m.start() < best_start):
            best_start, best_end, best_repl = m.start(), m.end(), repl
    if best_start is None:
        return text
    return text[:best_start] + best_repl + text[best_end:]


def _fix_duplicate_actor_names(text: str, name: str, gender: Optional[str]) -> str:
    """Collapse 'Name and Name' when only one actor was identified."""
    other = _other_person_phrase(gender)
    escaped = re.escape(name)
    text = re.sub(
        rf"\b{escaped}\s+and\s+{escaped}\b",
        f"{name} and {other}",
        text,
        flags=re.IGNORECASE,
    )
    if gender in {"male", "female"}:
        text = re.sub(
            rf"\b{escaped}\s+and\s+another person\b",
            f"{name} and {other}",
            text,
            flags=re.IGNORECASE,
        )
    return text


# Gendered-attire cues: when a known-MALE actor's name sits right before female
# attire (or vice-versa), the model applied the name to the wrong person.
_FEMALE_ATTIRE = re.compile(
    r"\b(saree|sari|lehenga|ghagra|choli|blouse|dupatta|bindi|jhumka|jhumki|"
    r"mangalsutra|anklet|payal|nath|nose ring|maang tikka|bridal)\b",
    re.IGNORECASE,
)
_MALE_ATTIRE = re.compile(
    r"\b(sherwani|kurta pajama|dhoti|lungi|turban|pagri|safa)\b",
    re.IGNORECASE,
)


def _fix_gender_attire_mismatch(text: str, name: str, gender: Optional[str]) -> str:
    """Replace a name occurrence that is contradicted by gendered attire.

    e.g. male actor "Shah Rukh Khan dressed in a saree" -> "A woman dressed in a
    saree" (only that occurrence; correctly-gendered uses of the name are kept).
    Uses the actor's KNOWN gender (face classifier) + the attire cue in the text.
    """
    if gender not in {"male", "female"}:
        return text
    mismatch = _FEMALE_ATTIRE if gender == "male" else _MALE_ATTIRE
    other_cap = "A woman" if gender == "male" else "A man"
    other_low = "a woman" if gender == "male" else "a man"
    escaped = re.escape(name)
    out: List[str] = []
    last = 0
    for m in re.finditer(rf"\b{escaped}(?:'s)?\b", text):
        window = text[m.end(): m.end() + 60]
        if not mismatch.search(window):
            continue
        prefix = text[last:m.start()].rstrip()
        at_sentence_start = (m.start() == 0) or prefix.endswith((".", "!", "?", "\n")) or prefix == ""
        repl = other_cap if at_sentence_start else other_low
        if text[m.start():m.end()].endswith("'s"):
            repl += "'s"
        out.append(text[last:m.start()])
        out.append(repl)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def best_faces_for_caption(actors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the highest-similarity face per actor slug."""
    best: Dict[str, Dict[str, Any]] = {}
    for actor in actors:
        if not isinstance(actor, dict):
            continue
        if actor.get("actor") in (None, "unknown"):
            continue
        slug = _slugify(str(actor.get("actor", "")))
        if not slug:
            continue
        sim = float(actor.get("similarity", 0) or 0)
        prev = best.get(slug)
        if prev is None or sim > float(prev.get("similarity", 0) or 0):
            best[slug] = actor
    return list(best.values())


def actors_for_caption_enforcement(rec: Dict[str, Any]) -> List[ActorRef]:
    """Actor list for s8 post-processing: top-confidence face per identified actor."""
    raw = rec.get("actors")
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        filtered = best_faces_for_caption(raw)
        if filtered:
            return filtered
    eligible = caption_eligible_actors(rec)
    return eligible

# Bucket prompt line that blocks using tagged actor names.
_NO_REAL_NAMES_LINE = re.compile(
    r"^\s*[-•]?\s*Do not name real people or actors\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def caption_eligible_actors(rec: Dict[str, Any]) -> List[str]:
    """Actor names safe to inject into s8 prompts/post-processing."""
    if rec.get("actor_status") != "tagged":
        return []
    return actor_display_names(rec.get("clip_actors") or rec.get("actors") or [])


def finalize_actor_status(rec: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> None:
    """Set actor_status from clip_actors and per-clip confidence scores."""
    cfg = cfg or {}
    names = rec.get("clip_actors") or []
    if not names:
        rec["actor_status"] = "no_match"
        return

    min_sim = float(rec.get("actor_tag_min_similarity", 0) or 0)
    min_margin = float(rec.get("actor_tag_min_margin", 0) or 0)
    caption_min_sim = float(cfg.get("actor_caption_min_similarity", 0.50))
    caption_min_margin = float(cfg.get("actor_caption_min_margin", 0.10))
    if min_sim >= caption_min_sim and min_margin >= caption_min_margin:
        rec["actor_status"] = "tagged"
    else:
        rec["actor_status"] = "low_confidence"


def actor_display_names(actors: List[ActorRef]) -> List[str]:
    """Accept clip_actors as name strings (s7) or tag dicts with display_name."""
    names: List[str] = []
    for a in actors:
        if isinstance(a, str):
            n = a.strip()
        else:
            n = (a.get("display_name") or "").strip()
            if not n and a.get("actor"):
                n = str(a["actor"]).replace("_", " ").title()
        if n and n not in names:
            names.append(n)
    return names


def strip_no_real_people_rule(bucket_prompt: str) -> str:
    return _NO_REAL_NAMES_LINE.sub("", bucket_prompt).strip()


def build_actor_caption_prompt(bucket_prompt: str, actors: List[ActorRef]) -> str:
    """Build VLM prompt: tagged actors must appear by name in every bullet."""
    names = actor_display_names(actors)
    if not names:
        return bucket_prompt

    base = strip_no_real_people_rule(bucket_prompt)
    if len(names) == 1:
        subject_rule = (
            f"The identified person is {names[0]}. "
            f"Use this full name once when describing them. "
            f"If other people are visible, call them 'another woman', 'another man', "
            f"or 'another person' — never reuse {names[0]}'s name for anyone else. "
            f"Do not write 'the man', 'the woman', or 'a person' for {names[0]}."
        )
        example = (
            f"• {names[0]}, wearing traditional attire, … • The setting is … "
            f"• The lighting … • The camera …"
        )
    else:
        joined = " and ".join(names)
        subject_rule = (
            f"Identified cast in this clip: {joined}. "
            "You must use each person's full name when describing them. "
            "Never write 'the man', 'the woman', 'the other person', or 'two individuals'."
        )
        example = (
            f"• {names[0]} and {names[1]} stand side by side; {names[0]} wears … "
            f"while {names[1]} wears … • The setting is … • … • …"
        )

    header = (
        "IDENTIFIED CAST (mandatory names when these people are visible):\n"
        f"{subject_rule}\n"
        f"Example opening: {example}\n\n"
    )
    return header + base


def enforce_actor_names_in_caption(
    caption: str,
    actors: List[ActorRef],
    *,
    gender_map: Dict[str, str] | None = None,
) -> str:
    """Replace generic people phrases with tagged display names."""
    names = actor_display_names(actors)
    if not caption or not names:
        return caption

    gmap = normalize_actor_gender_map(gender_map)
    male_names = [n for n in names if _gender_for_display_name(n, gmap) == "male"]
    female_names = [n for n in names if _gender_for_display_name(n, gmap) == "female"]

    pairs: List[tuple[str, str]] = []

    if len(names) == 1:
        n = names[0]
        g = _gender_for_display_name(n, gmap)
        other = _other_person_phrase(g)
        text = caption
        for pattern, repl in [
            (r"\btwo women\b", f"{n} and {other}"),
            (r"\bTwo women\b", f"{n} and {other}"),
            (r"\btwo men\b", f"{n} and {other}"),
            (r"\bTwo men\b", f"{n} and {other}"),
            (r"\btwo individuals\b", f"{n} and {other}"),
            (r"\bTwo individuals\b", f"{n} and {other}"),
            (r"\ba couple\b", f"{n} and {other}"),
            (r"\bA couple\b", f"{n} and {other}"),
        ]:
            text = re.sub(pattern, repl, text)
        text = _replace_first_person_reference(text, n, g)
        return _fix_duplicate_actor_names(text, n, g)
    elif len(male_names) == 1 and len(female_names) == 1:
        male, female = male_names[0], female_names[0]
        pair = f"{female} and {male}"
        pairs.extend(_man_replacement_patterns(male))
        pairs.extend(_woman_replacement_patterns(female))
        pairs.extend([
            (r"\bTwo women\b", pair),
            (r"\btwo women\b", pair),
            (r"\bBoth women\b", f"Both {female} and {male}"),
            (r"\bboth women\b", f"both {female} and {male}"),
            (r"\bThe other\b", male),
            (r"\bthe other\b", male),
            (r"\btwo individuals\b", pair),
            (r"\bTwo individuals\b", pair),
            (r"\ba couple\b", pair),
            (r"\bA couple\b", pair),
        ])
    elif len(names) >= 2:
        n0, n1 = names[0], names[1]
        pair = f"{n0} and {n1}"
        pairs.extend([
            (r"\bTwo women\b", pair),
            (r"\btwo women\b", pair),
            (r"\bBoth women\b", f"Both {n0} and {n1}"),
            (r"\bboth women\b", f"both {n0} and {n1}"),
            (r"\bOne wears\b", f"{n0} wears"),
            (r"\bone wears\b", f"{n0} wears"),
            (r"\bThe other\b", n1),
            (r"\bthe other\b", n1),
            (r"\bAnother woman\b", n1),
            (r"\banother woman\b", n1),
            (r"\btwo individuals\b", pair),
            (r"\bTwo individuals\b", pair),
            (r"\ba couple\b", pair),
            (r"\bA couple\b", pair),
        ])
        pairs.extend(_woman_replacement_patterns(n0))
        pairs.extend(_man_replacement_patterns(n1))

    text = caption
    for pattern, repl in pairs:
        text = re.sub(pattern, repl, text)
    return text


def _bbox_position(rec: Dict[str, Any], name: str) -> Optional[str]:
    """left/center/right of the detected actor's face from its bbox."""
    for a in rec.get("actors") or []:
        if isinstance(a, dict) and a.get("display_name") == name and a.get("bbox"):
            b = a["bbox"]
            if len(b) < 4:
                return None
            try:
                w = int(str(rec.get("crop_box") or "1920").split(":")[0])
            except (ValueError, IndexError):
                w = 1920
            frac = (float(b[0]) + float(b[2])) / 2.0 / max(1, w)
            return "left" if frac < 0.4 else "right" if frac > 0.6 else "center"
    return None


_POS_RE = re.compile(r"\b(left|right|cent(?:er|re)|middle)\b", re.IGNORECASE)
# Connector that introduces a SECOND, distinct person right before a name.
_SECOND_PERSON_LEAD = re.compile(
    r"(?:to (?:his|her) (?:left|right)|on (?:the|his|her) (?:left|right)|beside|"
    r"opposite|next to|alongside|facing|behind|in front of|approaches?|"
    r"with|and|the other (?:man|woman))\s*,?\s*$",
    re.IGNORECASE,
)


def _mention_position(text: str, start: int, end: int) -> Optional[str]:
    """Position label tied to one name mention (prefers a cue just before it)."""
    before = text[max(0, start - 34):start].lower()
    cue = None
    for m in _POS_RE.finditer(before):
        cue = m.group(1)
    if cue is None:
        m = _POS_RE.search(text[start:end + 22].lower())
        cue = m.group(1) if m else None
    if cue is None:
        return None
    cue = cue.lower()
    return "center" if cue in ("center", "centre", "middle") else cue


def _fix_same_gender_overtag(
    text: str, name: str, gender: Optional[str], rec: Dict[str, Any]
) -> str:
    """Same-gender over-tag: one actor detected but the name is applied to two
    people. Keep the name only on the mention matching the detected bbox
    position; rewrite the other distinct-person mention(s) to 'another man/woman'.
    Conservative: requires a known bbox position AND a clear 2-person structure
    AND at least one mention that matches the bbox position (the anchor)."""
    bpos = _bbox_position(rec, name)
    if bpos is None or gender not in {"male", "female"}:
        return text
    esc = re.escape(name)
    if not re.search(
        rf"(to (?:his|her) (?:left|right)|on the (?:left|right)|beside|opposite|"
        rf"with {esc}|and {esc}|the other (?:man|woman))",
        text,
        re.IGNORECASE,
    ):
        return text
    mentions = list(re.finditer(rf"\b{esc}\b", text))
    if len(mentions) < 2:
        return text
    poses = [_mention_position(text, m.start(), m.end()) for m in mentions]
    anchors = [i for i, p in enumerate(poses) if p == bpos]
    if not anchors:
        return text  # cannot anchor to the real person -> do nothing (safe)
    other_cap = "A woman" if gender == "female" else "A man"
    other_low = "a woman" if gender == "female" else "a man"
    rewrite = []
    for i, (m, p) in enumerate(zip(mentions, poses)):
        if i in anchors:
            continue
        lead = text[max(0, m.start() - 24):m.start()]
        if p is not None and p != bpos:
            rewrite.append(i)                       # explicit contradicting position
        elif _SECOND_PERSON_LEAD.search(lead):
            rewrite.append(i)                       # introduced as a distinct person
    for i in sorted(rewrite, reverse=True):
        m = mentions[i]
        prefix = text[:m.start()].rstrip()
        cap = (m.start() == 0) or prefix.endswith((".", "!", "?", "\n")) or prefix == ""
        text = text[:m.start()] + (other_cap if cap else other_low) + text[m.end():]
    return text


_OVERTAG_CONNECTOR = (
    r"(?:and|with|beside|opposite|facing|next to|alongside|"
    r"to (?:his|her) (?:left|right)|in front of|behind)"
)


def has_self_interaction_overtag(caption: str, rec: Dict[str, Any]) -> bool:
    """High-precision detector for same-gender over-tagging: the SAME actor
    name applied to two people interacting (you don't converse with yourself).
    e.g. 'Shah Rukh Khan ... conversation with Shah Rukh Khan'. Only fires for
    a single identified actor, so it can't misread a legit two-cast caption."""
    names = actor_display_names(actors_for_caption_enforcement(rec))
    if len(names) != 1:
        return False
    esc = re.escape(names[0])
    pat = rf"\b{esc}\b.{{0,90}}\b{_OVERTAG_CONNECTOR}\b.{{0,15}}\b{esc}\b"
    return re.search(pat, caption, re.IGNORECASE) is not None


def fix_actor_gender_tagging(
    caption: str,
    rec: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> str:
    """Targeted tagging post-fix: correct ONLY gender/duplicate actor mis-tags
    (name on wrong-gender person, 'Name and Name'). Unlike
    enforce_actor_names_for_record it does NOT force generic phrases to actor
    names, so the model's own caption style/quality is preserved.
    """
    names = actor_display_names(actors_for_caption_enforcement(rec))
    if not names:
        return caption
    gmap = actor_gender_map_from_config(config)
    text = caption
    for nm in names:
        g = _gender_for_actor_name(nm, rec, gmap)
        text = _fix_duplicate_actor_names(text, nm, g)
        text = _fix_gender_attire_mismatch(text, nm, g)
    # Safe grammar normalization of the model's awkward "the another X".
    text = re.sub(r"\bthe another\b", "the other", text)
    text = re.sub(r"\bThe another\b", "The other", text)
    # NOTE: same-gender over-tag (two men, one detected) is intentionally NOT
    # auto-fixed here. Text rules either miss it (wrong person is the subject,
    # no positional/gender cue) or corrupt correct multi-person captions. The
    # robust fix is detection coverage at s7 (match the 2nd person), not text
    # surgery. _fix_same_gender_overtag is kept for reference but unused.
    return text


def enforce_actor_names_for_record(
    caption: str,
    rec: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> str:
    actor_refs = actors_for_caption_enforcement(rec)
    names = actor_display_names(actor_refs)
    if not names:
        return caption
    text = enforce_actor_names_in_caption(
        caption,
        actor_refs,
        gender_map=actor_gender_map_from_config(config),
    )
    if len(names) == 1:
        gmap = actor_gender_map_from_config(config)
        g = _gender_for_actor_name(names[0], rec, gmap)
        text = _fix_duplicate_actor_names(text, names[0], g)
        text = _fix_gender_attire_mismatch(text, names[0], g)
    else:
        # multi-actor: fix each named actor against gendered-attire contradictions
        gmap = actor_gender_map_from_config(config)
        for nm in names:
            g = _gender_for_actor_name(nm, rec, gmap)
            text = _fix_gender_attire_mismatch(text, nm, g)
    return text
