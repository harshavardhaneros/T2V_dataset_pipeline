"""Actor-aware caption prompts and post-processing for s8."""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Bucket prompt line that blocks using tagged actor names.
_NO_REAL_NAMES_LINE = re.compile(
    r"^\s*[-•]?\s*Do not name real people or actors\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def actor_display_names(actors: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for a in actors:
        n = (a.get("display_name") or "").strip()
        if not n and a.get("actor"):
            n = str(a["actor"]).replace("_", " ").title()
        if n:
            names.append(n)
    return names


def strip_no_real_people_rule(bucket_prompt: str) -> str:
    return _NO_REAL_NAMES_LINE.sub("", bucket_prompt).strip()


def build_actor_caption_prompt(bucket_prompt: str, actors: List[Dict[str, Any]]) -> str:
    """Build VLM prompt: tagged actors must appear by name in every bullet."""
    names = actor_display_names(actors)
    if not names:
        return bucket_prompt

    base = strip_no_real_people_rule(bucket_prompt)
    if len(names) == 1:
        subject_rule = (
            f"The only person to name is {names[0]}. "
            f"Start bullet 1 with '{names[0]}' (not 'a woman', 'a man', or 'a person')."
        )
        example = (
            f"• {names[0]}, wearing traditional attire, … • The setting is … "
            f"• The lighting … • The camera …"
        )
    else:
        joined = " and ".join(names)
        subject_rule = (
            f"The people in frame are: {joined}. "
            f"Use these full names in bullet 1 (e.g. '{joined} stand side by side'). "
            f"Never write 'two women', 'both women', 'one woman', 'the other woman', or 'a woman'."
        )
        example = (
            f"• {names[0]} and {names[1]} stand side by side; {names[0]} wears … "
            f"while {names[1]} wears … • The setting is … • … • …"
        )

    header = (
        "IDENTIFIED PEOPLE (mandatory — overrides any rule below that forbids naming people):\n"
        f"{subject_rule}\n"
        f"Example opening: {example}\n\n"
    )
    return header + base


def enforce_actor_names_in_caption(caption: str, actors: List[Dict[str, Any]]) -> str:
    """Replace generic people phrases with tagged display names."""
    names = actor_display_names(actors)
    if not caption or not names:
        return caption

    text = caption
    if len(names) == 1:
        n = names[0]
        pairs = [
            (r"\bTwo women\b", n),
            (r"\btwo women\b", n),
            (r"\bA woman\b", n),
            (r"\ba woman\b", n),
            (r"\bThe woman\b", n),
            (r"\bthe woman\b", n),
            (r"\bA person\b", n),
            (r"\ba person\b", n),
            (r"\bThe person\b", n),
            (r"\bthe person\b", n),
        ]
    else:
        n0, n1 = names[0], names[1]
        pair = f"{n0} and {n1}"
        pairs = [
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
            (r"\bA woman\b", n0),
            (r"\ba woman\b", n0),
            (r"\bThe woman\b", n0),
            (r"\bthe woman\b", n0),
        ]

    for pattern, repl in pairs:
        text = re.sub(pattern, repl, text)
    return text
