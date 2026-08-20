#!/usr/bin/env bash
# Fires on EVERY user prompt. Injects Arc's operating rules into the turn's context,
# so they are present before the reply is written rather than only at session start.
# A rule in a file that is read once per session is not a mechanism.
cat <<'RULES'
<arc-operating-rules>
🔴 LOST / COMPACTED / NEW SESSION? READ THIS FILE FIRST, TOP TO BOTTOM, BEFORE REPLYING:
   ~/.claude/projects/-Users-jish-Documents-GitHub-arc/memory/mission/00_RESUME_HERE.md
   It carries current state, the map to x4-8, what is MEASURED vs RETRACTED, open
   defects + owners, the system tree, and the bounded read order. Do not reconstruct
   any of it from the conversation summary — the summary is lossy and has been wrong.

WHY ARC EXISTS: Runcrate rents GPUs. The wedge is CAPACITY PER NODE — one node serving
4-8x more multiplies a fleet without buying a card. x4-8 credible, ~x1 shipped. THE MOAT
IS THE BYTE FORMATS (trellis weights, compressed KV), not any one kernel — which is why
the GEMM and the attention kernel must be ours.

1. A SCOPING RESULT IS NEVER A VERDICT. "Doesn't work yet" / "lower ceiling than hoped" /
   "costs throughput here" => BUILD IT AND FIX IT. Never rank a novel system down, never
   turn one off as a conclusion. Report scope, not sentence. No limiting beliefs.
2. GET THE BOX, GET THE NUMBER, THEN OPEN THE PR. "Should help" is not finished work.
   CPU-only validation never substitutes for hardware.
3. A GREEN RESULT MUST PROVE WORK HAPPENED. Silent success is the house fault (13+
   instances). Assert engagement; environment failure exits 2 not 1; "no failures" is not
   "no results". Verification code is where this bug hides best.
4. DON'T RE-APOLOGISE. Fix the reflex, record it once, move on.
5. MAIN ORCHESTRATES — dispatch agents, don't do their work. Agents use
   ~/.config/arc/bin/arcgpu, never bare runcrate (single-use OAuth token).
6. LAYMAN BY DEFAULT with Jish. Describe what changed, never cite PR numbers as the
   answer. Report when a number lands, not when an agent breathes.

Full rules + provenance: memory/mission/{KERNEL_RULES.md (D16-D21), GPU_ACCESS_RULE.md
(D14-D15), FACTS.md (measured only), CEILINGS.json (physics vs implementation)}.
</arc-operating-rules>
RULES
