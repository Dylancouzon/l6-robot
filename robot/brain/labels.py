"""Turning a spoken sentence into the object's name."""
import re

# ASR usually drops the punctuation that would end the naming clause, so the
# label also stops at words that start a new one ("...my mug I bought it").
_CLAUSE_WORDS = {"i", "it", "and", "that", "which", "because", "she", "he",
                 "they", "we", "you", "made", "bought", "got", "from", "at",
                 "about", "in", "on"}


def parse_label(transcript):
    """'This is my mug, Maria made it.' -> 'my mug'.

    Takes the words after "this is", stops at the first word that starts a new
    clause, and keeps at most five of them, so a rambling sentence still gives
    a name short enough to fit on a box. Falls back to the whole transcript
    when nothing matches.

    Lowercased here, at the one door every label passes through: labels are
    compared as exact strings (forget deletes by label, recall dedupes by
    label) and Whisper capitalizes on a whim, so "Water bottle" and "water
    bottle" would otherwise be two objects.
    """
    m = re.search(r"this is (?:an? )?(.+?)(?:\s*[,.;!?—–-]|$)",
                  transcript, re.IGNORECASE)
    phrase = m.group(1) if m else transcript
    words = []
    for w in phrase.split():
        if w.lower().strip(".,!?") in _CLAUSE_WORDS:
            break
        words.append(w)
    return " ".join(words[:5]).rstrip(".,!?").lower() or "unnamed"



def norm_label(text):
    """The shape a TYPED label takes: collapsed whitespace, lowercased.

    Rename is the second door a label walks in through; `parse_label` above is
    the first, and it ends in the same lowercasing. They are separate on
    purpose - one parses a sentence, this normalizes a name - but a typed name
    and a spoken one have to end up the same string, because it is a key. Both
    callers here (the stored label, and the vector recall matches it by) go
    through this one, so those two can never disagree.
    """
    return " ".join(text.split()).lower()
