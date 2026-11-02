"""orbitcheck: exact checker for organism 4 (the true fixed point and the learner's attractor).

Reference process: state n in {0..63}, F(n) = floor(n/2); unique fixed point 0.
A trajectory is a list of packets. Packet = (numeral text, note text or None).
Everything here is deterministic and judge-free.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict

N = 64

def F(n: int) -> int:
    return n // 2

def reference_orbit(n: int, steps: int) -> list[int]:
    out = [n]
    for _ in range(steps):
        out.append(F(out[-1]))
    return out

# ---- encodings -----------------------------------------------------------
def encode(n: int, enc: str) -> str:
    if enc == "dec":
        return str(n)
    if enc == "bin":
        return format(n, "06b")
    raise ValueError(enc)

def decode(s: str, enc: str) -> int | None:
    s = s.strip().strip(".`*").strip()
    try:
        if enc == "dec" and re.fullmatch(r"\d{1,3}", s):
            v = int(s)
        elif enc == "bin" and re.fullmatch(r"[01]{1,8}", s):
            v = int(s, 2)
        else:
            return None
    except ValueError:
        return None
    return v

# ---- packet parsing -------------------------------------------------------
STATE_RE = re.compile(r"^\s*\**STATE\**\s*:\s*\**\s*([^\s*]+)", re.M | re.I)
NOTE_RE = re.compile(r"^\s*\**NOTE\**\s*:\s*\**\s*(.*)\Z", re.M | re.I | re.S)
REFUSAL_RE = re.compile(r"\b(I can(?:'|no)t|I cannot|I won't|I'm unable|I am unable|unable to comply|I must decline)\b", re.I)

def note_tokens(s: str) -> int:
    """Counting rule for the 64-token note bound: word and punctuation runs."""
    return len(re.findall(r"\w+|[^\w\s]", s))

@dataclass
class Parsed:
    status: str            # OK | INVALID | REFUSAL | OVERFLOW | OUT_OF_RANGE
    numeral: str | None
    value: int | None
    note: str | None
    note_tokens: int = 0

def parse_reply(text: str, enc: str, want_note: bool, note_cap: int = 64) -> Parsed:
    m = STATE_RE.search(text or "")
    if not m:
        st = "REFUSAL" if REFUSAL_RE.search(text or "") else "INVALID"
        return Parsed(st, None, None, None)
    numeral = m.group(1).strip().strip(".`*")
    v = decode(numeral, enc)
    note = None
    nt = 0
    if want_note:
        nm = NOTE_RE.search(text)
        if not nm:
            return Parsed("INVALID", numeral, v, None)
        note = nm.group(1).strip()
        nt = note_tokens(note)
        if nt > note_cap:
            return Parsed("OVERFLOW", numeral, v, note, nt)
    if v is None:
        return Parsed("INVALID", numeral, None, note, nt)
    if not (0 <= v < N):
        return Parsed("OUT_OF_RANGE", numeral, v, note, nt)
    return Parsed("OK", numeral, v, note, nt)

# ---- recurrence measures --------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def ned(a: str, b: str) -> float:
    """Normalised edit distance in [0,1]."""
    if not a and not b:
        return 0.0
    return edit_distance(a, b) / max(len(a), len(b))

def numeric_cycle(vals: list[int]) -> tuple[int, int] | None:
    """Eventual period of the tail of an integer sequence: (period, first index of the cycle)
    when the last value recurs; period 1 is a numeric fixed point. Requires the cycle to have
    been traversed at least twice (2*period trailing values agree)."""
    n = len(vals)
    for p in range(1, n // 2 + 1):
        if all(vals[n - 1 - k] == vals[n - 1 - k - p] for k in range(p)):
            start = n - 2 * p
            while start > 0 and vals[start - 1] == vals[start - 1 + p]:
                start -= 1
            return p, start
    return None

@dataclass
class Report:
    n_steps: int
    states: list
    status_final: str
    transitions_correct: int
    transitions_checked: int
    first_wrong_step: int | None
    reached_zero: bool
    persisted_at_zero: bool          # once at 0, never left
    left_zero: int                   # number of 0 -> nonzero transitions
    numeric_tail: tuple | None       # (period, start) of numeric tail
    wrong_fixed_point: int | None    # nonzero x with sustained x->x
    wrong_cycle: tuple | None        # (period, states) of a numeric cycle not equal to {0}
    packet_repeat: tuple | None      # (i, j): packet j identical to earlier packet i (first such)
    packet_period: int | None
    note_ned_consecutive: list = field(default_factory=list)
    note_min_ned_prior: list = field(default_factory=list)
    note_tokens: list = field(default_factory=list)
    invalid: int = 0
    refusals: int = 0
    overflow: int = 0
    recovered_after_error: bool | None = None  # after first wrong step, did it return to the reference fixed point 0?

def check(start_value: int, packets: list[tuple], statuses: list[str]) -> Report:
    """packets[k] = (value or None, note or None) for k = 0..T; packets[0] is the seeded first packet.
    statuses[k] is the parse status of packet k (packets[0] is 'SEED')."""
    vals = [p[0] for p in packets]
    ok_vals = []
    correct = checked = 0
    first_wrong = None
    for k in range(1, len(packets)):
        a, b = vals[k - 1], vals[k]
        if a is None or b is None:
            break
        checked += 1
        if b == F(a):
            correct += 1
        elif first_wrong is None:
            first_wrong = k
    for v in vals:
        if v is None:
            break
        ok_vals.append(v)
    reached = 0 in ok_vals
    z = ok_vals.index(0) if reached else None
    left = sum(1 for k in range(1, len(ok_vals)) if ok_vals[k - 1] == 0 and ok_vals[k] != 0)
    persisted = reached and all(v == 0 for v in ok_vals[z:])
    tail = numeric_cycle(ok_vals) if len(ok_vals) >= 3 else None
    wfp = wcyc = None
    if tail:
        p, s = tail
        cyc = ok_vals[s:s + p]
        if p == 1 and cyc[0] != 0:
            wfp = cyc[0]
        elif p > 1:
            wcyc = (p, cyc)
    # exact full-packet recurrence (value AND note text)
    rep = per = None
    seen = {}
    for k, pk in enumerate(packets):
        key = (pk[0], pk[1])
        if key in seen and pk[0] is not None:
            rep = (seen[key], k); per = k - seen[key]
            break
        seen[key] = k
    notes = [pk[1] for pk in packets if pk[1] is not None]
    cons = [round(ned(notes[k - 1], notes[k]), 3) for k in range(1, len(notes))]
    mprior = [round(min(ned(notes[j], notes[k]) for j in range(k)), 3) for k in range(1, len(notes))]
    rec = None
    if first_wrong is not None:
        rec = reached and ok_vals[-1] == 0
    return Report(
        n_steps=len(packets) - 1, states=vals, status_final=statuses[-1],
        transitions_correct=correct, transitions_checked=checked, first_wrong_step=first_wrong,
        reached_zero=reached, persisted_at_zero=persisted, left_zero=left,
        numeric_tail=tail, wrong_fixed_point=wfp, wrong_cycle=wcyc,
        packet_repeat=rep, packet_period=per,
        note_ned_consecutive=cons, note_min_ned_prior=mprior,
        note_tokens=[note_tokens(n) for n in notes],
        invalid=statuses.count("INVALID") + statuses.count("OUT_OF_RANGE"),
        refusals=statuses.count("REFUSAL"), overflow=statuses.count("OVERFLOW"),
        recovered_after_error=rec)

def as_dict(r: Report) -> dict:
    return asdict(r)

def _selftest():
    assert reference_orbit(13, 5) == [13, 6, 3, 1, 0, 0]
    assert reference_orbit(7, 4) == [7, 3, 1, 0, 0]
    assert all(reference_orbit(n, 7)[-1] == 0 for n in range(N))
    assert decode("001101", "bin") == 13 and encode(13, "bin") == "001101"
    p = parse_reply("STATE: 6\nNOTE: halved it", "dec", True)
    assert p.status == "OK" and p.value == 6 and p.note == "halved it"
    assert parse_reply("STATE: 6\nNOTE: " + "w " * 70, "dec", True).status == "OVERFLOW"
    assert parse_reply("I cannot do that.", "dec", False).status == "REFUSAL"
    assert numeric_cycle([5, 3, 4, 3, 4, 3, 4]) == (2, 1)
    assert numeric_cycle([13, 6, 3, 1, 0, 0, 0]) == (1, 4)
    r = check(13, [(7, None), (3, None), (1, None), (0, None), (0, None)], ["SEED"] + ["OK"] * 4)
    assert r.reached_zero and r.persisted_at_zero and r.first_wrong_step is None
    r = check(13, [(7, "a"), (5, "b"), (5, "b"), (5, "b")], ["SEED"] + ["OK"] * 3)
    assert r.wrong_fixed_point == 5 and r.packet_repeat == (1, 2) and r.first_wrong_step == 1
    assert edit_distance("kitten", "sitting") == 3
    print("orbitcheck selftest ok")

if __name__ == "__main__":
    _selftest()
