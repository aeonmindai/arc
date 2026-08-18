# Engineering record

The measurements, negative results, and standing decisions behind Arc's
quantization and performance work — kept in the repository so they are versioned,
shared, and survive independently of any one person's notes.

This is not a tutorial. It is the evidence trail for claims Arc makes, written so
that a competent engineer who has never seen this codebase can check the
arithmetic, find the source, and tell a measurement from a projection.

| document | what is in it |
|---|---|
| [QUANTIZATION_PERFORMANCE.md](QUANTIZATION_PERFORMANCE.md) | Where the trellis bake's time goes, what was optimized, the measured architectural ceiling, and the four predictions that failed |
| [HARDWARE_LESSONS.md](HARDWARE_LESSONS.md) | What it costs to run this on rented GPUs: driver/toolkit gates, bad-box detection, the `cudnn` regression, thread policy, and the bake OOM |
| [TESTING_DISCIPLINE.md](TESTING_DISCIPLINE.md) | Seven tests found passing while verifying nothing, the mechanism of each, and the practices adopted in response — this is the charter of the **ArcGate** system |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | What is deferred and why, and the honest list of what has not been measured |

Arc's subsystems are organised under named parent systems (ArcInfer, ArcQuant,
ArcKernels, ArcGate, …). The full tree, including what is shipped versus planned
versus nonexistent, is in
[`memory/mission/TAXONOMY.md`](../../memory/mission/TAXONOMY.md).

## Evidence grades

Every number in these documents carries one of these labels. A number without one
is a defect in the document.

| grade | meaning |
|---|---|
| **[measured]** | someone ran it on hardware; the box and the shape are stated |
| **[derived]** | arithmetic over measured quantities or over constants read in shipped source |
| **[source-verified]** | read directly in shipped code, ours or a third party's |
| **[projected]** | a forward estimate — not a measurement |
| **[published]** | a third party's number, measured against *their* baseline, not ours |

**A projection is never presented as a measurement.** Where a projection was later
measured and found wrong, both numbers are kept and the mechanism of the error is
explained — those entries have been the most useful pages here.
