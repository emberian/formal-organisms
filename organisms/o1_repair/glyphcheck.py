"""glyphcheck: the checker for organism o1 (first error, human correction, delayed reconstruction).

A fresh-glyph Boolean operator algebra: a spec maps glyphs to truth tables
(binary glyphs written infix, unary glyphs prefix). A derivation is a list of
nodes `dK: <expr> = v` whose expressions may reference inputs and earlier
nodes. The checker evaluates expressions, checks every node locally (against
the claimed values of the nodes it references) and globally (against the true
values), tracks the dependency DAG, returns the earliest invalid node with
every affected descendant, and scores a proposed repair.

Verdicts follow the shared conventions: VALID, INVALID(counterexample),
UNRESOLVED (unparseable or incomplete input). Nothing here is a judge model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# --- connectives -------------------------------------------------------------
# Binary tables are keyed (x, y) -> value, in row order 00, 01, 10, 11.
def _t(s: str) -> dict:
    return {(0, 0): int(s[0]), (0, 1): int(s[1]), (1, 0): int(s[2]), (1, 1): int(s[3])}

BINARY = {
    "AND": _t("0001"), "OR": _t("0111"), "XOR": _t("0110"), "XNOR": _t("1001"),
    "NAND": _t("1110"), "NOR": _t("1000"), "IMP": _t("1101"), "RIMP": _t("1011"),
    "NIMP": _t("0010"), "NRIMP": _t("0100"),
}
UNARY = {"NOT": {0: 1, 1: 0}}


@dataclass
class Spec:
    """glyph -> connective name; `override` lets a reader substitute a table."""
    binary: dict[str, str]
    unary: dict[str, str]

    def table(self, glyph: str, override: dict | None = None):
        if override and glyph in override:
            return BINARY[override[glyph]]
        if glyph in self.binary:
            return BINARY[self.binary[glyph]]
        return UNARY[self.unary[glyph]]

    def render(self) -> str:
        out = ["Values: 0, 1", ""]
        for g, name in self.binary.items():
            out.append(f"a {g} b:")
            for (x, y), v in BINARY[name].items():
                out.append(f"  {x} {g} {y} = {v}")
            out.append("")
        for g, name in self.unary.items():
            for x, v in UNARY[name].items():
                out.append(f"{g}{x} = {v}")
            out.append("")
        return "\n".join(out).rstrip() + "\n"


# --- expressions -------------------------------------------------------------
# AST: ("lit", v) | ("var", name) | ("un", g, e) | ("bin", g, l, r)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class ParseError(ValueError):
    pass


def tokenize(s: str, spec: Spec) -> list[str]:
    glyphs = sorted(list(spec.binary) + list(spec.unary), key=len, reverse=True)
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()01":
            toks.append(c); i += 1; continue
        m = _IDENT.match(s, i)
        if m:
            toks.append(m.group()); i = m.end(); continue
        for g in glyphs:
            if s.startswith(g, i):
                toks.append(g); i += len(g); break
        else:
            # tolerate variation selectors that some models append to glyphs
            if c in "︎️":
                i += 1; continue
            raise ParseError(f"unexpected character {c!r} at {i} in {s!r}")
    return toks


def parse(s: str, spec: Spec):
    """Binary glyphs are left-associative with one precedence level; unary binds tightest."""
    toks = tokenize(s, spec)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def atom():
        nonlocal pos
        t = peek()
        if t is None:
            raise ParseError(f"unexpected end in {s!r}")
        if t in spec.unary:
            pos += 1
            return ("un", t, atom())
        if t == "(":
            pos += 1
            e = expr()
            if peek() != ")":
                raise ParseError(f"missing ) in {s!r}")
            pos += 1
            return e
        if t in ("0", "1"):
            pos += 1
            return ("lit", int(t))
        if _IDENT.fullmatch(t):
            pos += 1
            return ("var", t)
        raise ParseError(f"unexpected token {t!r} in {s!r}")

    def expr():
        nonlocal pos
        e = atom()
        while peek() in spec.binary:
            g = toks[pos]; pos += 1
            e = ("bin", g, e, atom())
        return e

    e = expr()
    if pos != len(toks):
        raise ParseError(f"trailing tokens {toks[pos:]} in {s!r}")
    return e


def evaluate(ast, env: dict, spec: Spec, override: dict | None = None) -> int:
    k = ast[0]
    if k == "lit":
        return ast[1]
    if k == "var":
        if ast[1] not in env:
            raise KeyError(ast[1])
        return env[ast[1]]
    if k == "un":
        return spec.table(ast[1])[evaluate(ast[2], env, spec, override)]
    l = evaluate(ast[2], env, spec, override)
    r = evaluate(ast[3], env, spec, override)
    return spec.table(ast[1], override)[(l, r)]


def refs(ast) -> set[str]:
    k = ast[0]
    if k == "var":
        return {ast[1]}
    if k == "un":
        return refs(ast[2])
    if k == "bin":
        return refs(ast[2]) | refs(ast[3])
    return set()


def render(ast) -> str:
    k = ast[0]
    if k == "lit":
        return str(ast[1])
    if k == "var":
        return ast[1]
    if k == "un":
        inner = render(ast[2])
        return f"{ast[1]}{inner}" if ast[2][0] in ("lit", "var", "un") else f"{ast[1]}({inner})"
    l = render(ast[2]) if ast[2][0] != "bin" else f"({render(ast[2])})"
    r = render(ast[3]) if ast[3][0] != "bin" else f"({render(ast[3])})"
    return f"{l} {ast[1]} {r}"


def glyph_uses(ast, env, spec, override=None) -> list[tuple]:
    """Every binary application as (glyph, x, y) with operand values (true semantics)."""
    out = []
    def go(a):
        if a[0] == "un":
            go(a[2])
        elif a[0] == "bin":
            go(a[2]); go(a[3])
            out.append((a[1], evaluate(a[2], env, spec, override), evaluate(a[3], env, spec, override)))
    go(ast)
    return out


# --- derivations -------------------------------------------------------------
@dataclass
class Node:
    id: str
    expr: str
    value: int


@dataclass
class Derivation:
    inputs: dict[str, int]
    nodes: list[Node]

    def render(self) -> str:
        ins = ", ".join(f"{k} = {v}" for k, v in self.inputs.items())
        w = max(len(n.expr) for n in self.nodes)
        return "\n".join([f"Inputs: {ins}", ""] +
                         [f"{n.id}: {n.expr.ljust(w)} = {n.value}" for n in self.nodes])


_NODE_LINE = re.compile(r"^\s*(\w+)\s*:\s*(.+?)\s*=\s*([01])\s*$")


def parse_derivation(text: str) -> Derivation:
    inputs, nodes = {}, []
    for line in text.splitlines():
        if line.strip().lower().startswith("inputs:"):
            for part in line.split(":", 1)[1].split(","):
                k, v = part.split("=")
                inputs[k.strip()] = int(v.strip())
            continue
        m = _NODE_LINE.match(line)
        if m:
            nodes.append(Node(m.group(1), m.group(2), int(m.group(3))))
    return Derivation(inputs, nodes)


@dataclass
class Check:
    true_values: dict[str, int]
    local_invalid: list[str]          # node wrong given the claimed values of its references
    global_invalid: list[str]         # node claimed value != true value
    earliest_invalid: str | None
    affected: list[str]               # transitive dependents of earliest_invalid (excluding it)
    deps: dict[str, set[str]]         # direct node references
    counterexample: str | None = None

    @property
    def verdict(self) -> str:
        return "VALID" if self.earliest_invalid is None else f"INVALID({self.counterexample})"


def dependents(deps: dict[str, set[str]], root: str, order: list[str]) -> list[str]:
    cone = {root}
    for nid in order:
        if deps[nid] & cone:
            cone.add(nid)
    return [n for n in order if n in cone and n != root]


def check(der: Derivation, spec: Spec) -> Check:
    order = [n.id for n in der.nodes]
    asts = {n.id: parse(n.expr, spec) for n in der.nodes}
    deps = {nid: refs(asts[nid]) & set(order) for nid in order}
    true_env, claim_env = dict(der.inputs), dict(der.inputs)
    local_bad, global_bad, cex = [], [], None
    for n in der.nodes:
        true_env[n.id] = evaluate(asts[n.id], true_env, spec)
        local = evaluate(asts[n.id], claim_env, spec)
        if local != n.value:
            local_bad.append(n.id)
            if cex is None:
                sub = render(_subst(asts[n.id], claim_env))
                cex = f"{n.id}: {sub} = {local}, not {n.value}"
        if true_env[n.id] != n.value:
            global_bad.append(n.id)
        claim_env[n.id] = n.value
    earliest = local_bad[0] if local_bad else None
    affected = dependents(deps, earliest, order) if earliest else []
    tv = {k: v for k, v in true_env.items() if k in deps}
    return Check(tv, local_bad, global_bad, earliest, affected, deps, cex)


def _subst(ast, env):
    k = ast[0]
    if k == "var":
        return ("lit", env[ast[1]]) if ast[1] in env else ast
    if k == "un":
        return ("un", ast[1], _subst(ast[2], env))
    if k == "bin":
        return ("bin", ast[1], _subst(ast[2], env), _subst(ast[3], env))
    return ast


# --- repair scoring ----------------------------------------------------------
def extract_json(text: str):
    """First balanced top-level JSON object in the text (fenced or not); None if absent."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    cands.append(text[start:i + 1]); break
        start = text.find("{", start + 1) if not cands else -1
    for c in cands:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def score_work(work: str, value: int, spec: Spec) -> bool:
    """`work` is the node's expression with operand values substituted (literals only)."""
    try:
        ast = parse(work, spec)
        return not refs(ast) and evaluate(ast, {}, spec) == value
    except (ParseError, KeyError):
        return False


def score_repair(obj, inherited: Derivation, spec: Spec) -> dict:
    """Score a repair artifact {"diagnosis","contradiction","nodes":[{id,value,work}],"explanation"}."""
    chk = check(inherited, spec)
    truth = chk.true_values
    order = [n.id for n in inherited.nodes]
    old = {n.id: n.value for n in inherited.nodes}
    exprs = {n.id: parse(n.expr, spec) for n in inherited.nodes}
    res = {"parsed": obj is not None and isinstance(obj.get("nodes"), list) if isinstance(obj, dict) else False}
    if not res["parsed"]:
        res.update(verdict="UNRESOLVED", repair_valid=False)
        return res
    got = {}
    for e in obj["nodes"]:
        try:
            got[str(e["id"])] = (int(e["value"]), str(e.get("work", "")).strip())
        except (KeyError, TypeError, ValueError):
            continue
    missing = [n for n in order if n not in got]
    wrong = [n for n in order if n in got and got[n][0] != truth[n]]
    cone = set(([chk.earliest_invalid] if chk.earliest_invalid else []) + chk.affected)
    per = {}
    for nid in order:
        if nid not in got:
            per[nid] = "missing"; continue
        v, w = got[nid]
        if w.lower() in ("kept", "unchanged", ""):
            per[nid] = "kept_changed" if v != old[nid] else "kept"
        else:
            # work must use the proposed values of the node's references
            ref_env = {r: got[r][0] for r in chk.deps[nid] if r in got}
            expect = render(_subst(exprs[nid], {**inherited.inputs, **ref_env}))
            ok = score_work(w, v, spec) and _same_literals(w, expect, spec)
            per[nid] = ("rederived" if ok else "work_invalid") + ("_revised" if v != old[nid] else "")
    diag = str(obj.get("diagnosis", "")).strip().upper()
    res.update(
        verdict="VALID" if not missing and not wrong else
                f"INVALID({(missing or wrong)[0]}: claimed {got.get((missing or wrong)[0], ('missing',))[0]}, true {truth[(missing or wrong)[0]]})",
        repair_valid=not missing and not wrong, missing=missing, wrong=wrong, per_node=per,
        cone=sorted(cone, key=order.index),
        cone_rederived=sum(per[n].startswith("rederived") for n in cone),
        cone_kept=sum(per[n].startswith("kept") for n in cone),
        outside_rederived=sum(per[n].startswith("rederived") for n in order if n not in cone),
        outside_kept=sum(per[n].startswith("kept") for n in order if n not in cone),
        revised=[n for n in order if n in got and got[n][0] != old[n]],
        diagnosis=diag,
        spec_inconsistent_claim=diag == "SPECIFICATION_INCONSISTENT",
        spec_inconsistent_supported=diag == "SPECIFICATION_INCONSISTENT" and contradiction_valid(obj.get("contradiction"), spec),
    )
    return res


def _same_literals(work: str, expect: str, spec: Spec) -> bool:
    try:
        return parse(work, spec) == parse(expect, spec)
    except ParseError:
        return False


def contradiction_valid(c, spec: Spec) -> bool:
    """A contradiction claim is valid only if the spec gives one glyph two values on the same row.
    Specs built here are total functions, so a claim passes only if the checker finds that conflict."""
    try:
        g, x, y = c["glyph"], int(c["inputs"][0]), int(c["inputs"][1])
        vals = {int(v) for v in c["values"]}
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    if g not in spec.binary or len(vals) < 2:
        return False
    return all(v == spec.table(g)[(x, y)] for v in vals)  # two distinct values cannot both be the table's


def score_answers(text: str, items: list[dict], spec: Spec, sub: dict) -> list[dict]:
    """Batch answers `eK = v` (last occurrence wins). `sub` is the planted substitution override."""
    found = {}
    for m in re.finditer(r"^\s*\**(e\d+)\**\s*[:=]\s*\**([01])\**\s*$", text, re.M):
        found[m.group(1)] = int(m.group(2))
    out = []
    for it in items:
        ast = parse(it["expr"], spec)
        t, w = evaluate(ast, {}, spec), evaluate(ast, {}, spec, sub)
        a = found.get(it["id"])
        out.append({"id": it["id"], "kind": it["kind"], "answer": a, "true": t, "sub_value": w,
                    "correct": a == t, "relapse_consistent": a is not None and a != t and a == w and t != w})
    return out
