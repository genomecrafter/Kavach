"""KAVACH guard: the plug-in layer that wraps any LLM.

    from kavach import Kavach, OpenAICompatLLM
    llm   = OpenAICompatLLM.from_preset("groq")              # the model you protect
    guard = Kavach(llm=llm, judge_llm=llm)                   # judge can be a different/cheaper model
    reply, decision = guard.chat([{"role": "user", "content": "..."}])
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .harm import assess_harm
from .judge import IntentJudge, JudgeVerdict
from .memory import AttackMemory, ConversationRisk
from .normalize import canonicalize
from .policy import (BLOCK_NOTE, GUARD_SYSTEM, REFUSALS, REFUSING, SANDWICH_REMINDER, SUPPORT, Action,
                     decide, lang_key)
from .tactics import detect_tactics


@dataclass
class Decision:
    action: Action
    harm: float
    tactic: float
    risk: float
    category: str | None
    language: str
    judge_called: bool
    forwarded_prompt: str | None
    trace: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d["action"] = self.action.value
        return d


class Kavach:
    def __init__(self, llm=None, judge_llm=None, mode: str = "cascade", memory_path: str | None = None,
                 audit_outputs: bool = True, fail_closed: bool = True, router=None, router_mode: str = "union"):
        """mode: 'heuristic' (no LLM judge), 'cascade' (judge only when uncertain), 'judge' (always judge).
        router: optional LearnedRouter. router_mode: 'union' (escalate if rules OR model say so),
        'learned' (model only, plus hard signals such as decoded payloads), 'rules' (ignore the model)."""
        self.llm = llm
        self.judge = IntentJudge(judge_llm) if judge_llm is not None else None
        self.mode = mode if self.judge else "heuristic"
        self.memory = AttackMemory(memory_path)
        self.conv = ConversationRisk()
        self.audit_outputs, self.fail_closed = audit_outputs, fail_closed
        self.router, self.router_mode = router, router_mode
        self.stats = {"requests": 0, "judge_calls": 0}

    # ------------------------------------------------------------------ input
    def analyze(self, text: str, session: str = "default") -> Decision:
        t0 = time.perf_counter()
        self.stats["requests"] += 1
        c = canonicalize(text)
        tactic, hits = detect_tactics(c)
        ha = assess_harm(c)
        mem = self.memory.similarity(c.analysis_text)
        conv_prev = self.conv.get(session)

        harm = ha.score
        category = max(ha.categories, key=ha.categories.get) if ha.categories else None
        if "self_harm" in ha.categories:   # first-person distress takes priority
            category = "self_harm"
        if mem >= self.memory.threshold:           # adaptive immunity
            harm = max(harm, 0.6)
            tactic = max(tactic, 0.5)
        lang = c.lang_hint

        letters = sum(c.scripts.values()) or 1
        unsupported_script = c.scripts.get("other", 0) / letters > 0.3   # e.g. Tamil/Telugu/Bengali: defer to judge
        rules_escalate = harm < 0.85 and (tactic >= 0.50 or (tactic >= 0.30 and harm > 0) or harm >= 0.15
                                          or mem >= 0.35 or conv_prev >= 0.6 or bool(c.decoded) or unsupported_script)
        p_router, router_error = None, None
        if self.router is not None and self.router_mode != "rules":
            try:
                p_router = round(float(self.router(text, self.conv.turns(session))), 4)
            except Exception as e:      # never let the router take the guard down: fall back to rules
                router_error = f"{type(e).__name__}: {e}"
        if p_router is not None:
            learned = p_router >= self.router.threshold
            hard = bool(c.decoded) or mem >= 0.35 or conv_prev >= 0.6
            escalate = harm < 0.85 and (learned or hard or (rules_escalate if self.router_mode == "union" else False))
        else:
            escalate = rules_escalate
        need_judge = self.mode == "judge" or (self.mode == "cascade" and escalate)
        verdict: JudgeVerdict | None = None
        if need_judge:
            self.stats["judge_calls"] += 1
            verdict = self.judge.judge(text, hints={
                "tactics": [h.tactic for h in hits], "topics": list(ha.categories),
                "decoded_payloads": c.decoded[:2], "romanized": c.roman[:600]},
                history=self.conv.turns(session))
            if verdict.error is None:
                harm = round(0.8 * verdict.risk + 0.2 * harm, 4)
                if verdict.tactics:
                    tactic = max(tactic, min(0.3 + 0.15 * len(verdict.tactics), 0.9))
                if verdict.harm_category not in ("none", "", None):
                    category = verdict.harm_category
                lang = verdict.user_language or lang
            elif self.fail_closed and harm >= 0.3:
                harm = max(harm, 0.6)

        # multi-turn escalation: carry part of earlier risk forward
        cum = self.conv.update(session, harm, text)
        if conv_prev >= 0.6 and harm >= 0.2:
            harm = max(harm, min(0.6, cum / 2))
            tactic = max(tactic, 0.45)
            hits_extra = ["multi_turn_escalation"]
        else:
            hits_extra = []

        rewrite = verdict.safe_rewrite if (verdict and verdict.error is None) else ""
        action = decide(harm, tactic, category, bool(rewrite))
        if action == Action.BLOCK and verdict is not None and verdict.error is None:
            self.memory.add(c.analysis_text)        # only judge-confirmed attacks are memorised

        forwarded = {Action.ALLOW: text, Action.GUARD: text, Action.REDIRECT: rewrite or text}.get(action)
        trace = {
            "escalate": escalate, "rules_escalate": rules_escalate, "router_prob": p_router, "router_error": router_error,
            "router_name": (getattr(self.router, "name", None) if self.router is not None else "rules"),
            "router_threshold": (self.router.threshold if self.router is not None else None),
            "unsupported_script": unsupported_script,
            "canonical": {"roman": c.roman[:400], "decoded": c.decoded, "signals": c.signals, "scripts": c.scripts},
            "tactics": [asdict(h) for h in hits] + [{"tactic": x, "weight": None, "evidence": "session"} for x in hits_extra],
            "harm_prefilter": asdict(ha), "memory_similarity": round(mem, 3), "conversation_risk": round(cum, 3),
            "judge": (asdict(verdict) if verdict else None),
        }
        return Decision(action, round(harm, 4), round(tactic, 4), round(min(1.0, harm * (1 + 0.5 * tactic)), 4),
                        category, lang, verdict is not None, forwarded, trace,
                        round((time.perf_counter() - t0) * 1000, 1))

    # --------------------------------------------------------------- full turn
    def refusal_text(self, d: Decision) -> str:
        k = lang_key(d.language)
        if d.action == Action.SUPPORT:
            return SUPPORT[k]
        msg = REFUSALS[k]
        return msg + BLOCK_NOTE[k] if d.action == Action.BLOCK else msg

    def chat(self, messages: list[dict], session: str = "default", llm_call: Callable | None = None, **kw):
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        d = self.analyze(user_text, session)
        if d.action in REFUSING or d.action == Action.SUPPORT:
            return self.refusal_text(d), d

        call = llm_call or (self.llm.chat if self.llm else None)
        if call is None:
            raise RuntimeError("No protected LLM configured.")
        msgs = [dict(m) for m in messages]
        if d.action in (Action.GUARD, Action.REDIRECT):
            msgs.insert(0, {"role": "system", "content": GUARD_SYSTEM})
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = d.forwarded_prompt + "\n\n" + SANDWICH_REMINDER
                    break
        reply = call(msgs, **kw)

        # output-side check: catches jailbreaks that slipped past the input stage
        if self.audit_outputs and d.action != Action.ALLOW:
            unsafe, why = (self.judge.audit_output(reply) if self.judge else self._heuristic_output_check(reply))
            d.trace["output_audit"] = {"unsafe": unsafe, "reason": why}
            if unsafe:
                d.action = Action.SAFE_REFUSE
                return self.refusal_text(d), d
        return reply, d

    @staticmethod
    def _heuristic_output_check(reply: str) -> tuple[bool, str]:
        c = canonicalize(reply)
        ha = assess_harm(c)
        unsafe = ha.score >= 0.75 and bool(ha.operational) and not ha.protective
        return unsafe, "lexicon" if unsafe else ""

    def protect(self, fn: Callable):
        """Decorator: @guard.protect on any fn(messages, **kw) -> str."""
        def wrapped(messages, session: str = "default", **kw):
            reply, _ = self.chat(messages, session=session, llm_call=fn, **kw)
            return reply
        return wrapped
