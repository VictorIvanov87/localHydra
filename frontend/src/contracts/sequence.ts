// Client-side SSE de-duplication, mirroring backend contracts/sse.py SequenceTracker.
// `gap` means: do not apply, reconnect with Last-Event-ID = tracker.last.
export type Accept = "apply" | "duplicate" | "gap";

export class SequenceTracker {
  last = 0;

  accept(sequence: number): Accept {
    if (sequence <= this.last) return "duplicate";
    if (sequence === this.last + 1) {
      this.last = sequence;
      return "apply";
    }
    return "gap";
  }
}
