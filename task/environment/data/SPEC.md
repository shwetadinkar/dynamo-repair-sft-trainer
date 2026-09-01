# sfttrainer — behavioural specification

This document is normative. Where the implementation in `/app/sfttrainer` disagrees with
this document, the implementation is wrong.

All arithmetic is float64 unless stated otherwise. All integer arrays are `int64`.

---

## 1. Entry point

    sfttrainer.pipeline.build_run(conversations, config) -> dict

`conversations` is a list of conversation objects (§2). `config` is a mapping with the
keys in §3. The return value is described in §9.

`build_run` is pure: the same arguments produce the same result, and neither argument is
mutated.

---

## 2. Conversation objects

    {"id": "<string>", "turns": [{"role": "user" | "assistant", "text": "<string>"}, ...]}

`id` is unique within a call. `turns` is non-empty and ordered. No other roles occur.
A conversation may contain any number of turns in any role order. Every turn's `text`
holds at least one content token, and content tokens are separated by single spaces
(U+0020); no other whitespace character occurs in a conversation.

---

## 3. Configuration

| key | type | meaning |
|---|---|---|
| `max_seq_len` | int > 0 | maximum token length of a single example (§6) |
| `max_tokens_per_microbatch` | int > 0 | padded token budget of one micro-batch (§7) |
| `grad_accum_steps` | int > 0 | micro-batches per optimizer step (§8) |
| `base_lr` | float > 0 | peak learning rate (§10) |
| `min_lr` | float ≥ 0 | floor learning rate (§10) |
| `warmup_steps` | int ≥ 0 | linear warmup length in optimizer steps (§10) |

Precondition, guaranteed by the caller: `max_tokens_per_microbatch >= max_seq_len`.

---

## 4. Vocabulary and tokenization

Vocabulary size is 50000. Reserved ids:

| id | token |
|---|---|
| 0 | `<pad>` |
| 1 | `<user>` |
| 2 | `<assistant>` |
| 3 | `<end>` |

A turn's `text` is split into content tokens on runs of whitespace, discarding empty
pieces; per §2 the only whitespace present is the single space. A content token string `s` has id

    4 + (zlib.crc32(s.encode("utf-8")) % 49996)

so content ids lie in `[4, 49999]` and never collide with reserved ids. Tokenization is
byte-exact and case-sensitive; no normalization, stripping, or lowercasing is applied.

---

## 5. Example assembly

A conversation is rendered to a token sequence by concatenating, for each turn in order:

1. the role marker — `<user>` (1) for a user turn, `<assistant>` (2) for an assistant turn;
2. the turn's content tokens, in order;
3. `<end>` (3).

The supervision mask has the same length as the token sequence. A position is supervised
(mask 1) if and only if it is a content token of an **assistant** turn, or the `<end>`
token closing an **assistant** turn. Every other position is 0 — all user positions, and
every role marker including `<assistant>` itself.

---

## 6. Truncation

If the rendered sequence is longer than `max_seq_len`, the token array and the mask array
are both truncated to their **first** `max_seq_len` entries.

After truncation, an example whose mask contains no `1` is dropped. It takes no part in
§7–§10 and does not appear in the output. A conversation containing no assistant turn is
dropped by this rule.

---

## 7. Bucketing and packing

Surviving examples are ordered by token length ascending, ties broken by `id` in
ascending lexicographic order. This ordering is the packing order and does not
depend on input order.

Micro-batches are formed greedily over that ordering. Let `B` be the micro-batch under
construction, `n` its current member count, and `Lmax` the greatest token length among
its current members (`Lmax = 0` when `B` is empty). A candidate example of length `Lc`
joins `B` if and only if

    (n + 1) * max(Lmax, Lc) <= max_tokens_per_microbatch

The width of the candidate is included in the capacity test. If the test fails, `B` is
closed and the candidate opens a new micro-batch. The test applies only to a non-empty
`B`; a candidate always joins an empty micro-batch. The precondition in §3 guarantees a
single example always fits.

A closed micro-batch of `n` members is materialized as two `(n, L)` arrays, where `L` is
the greatest token length among its members. Rows follow packing order. Each row is
right-padded to `L` with `<pad>` (0) in the token array and `0` in the mask array. The
padded size `n * L` never exceeds `max_tokens_per_microbatch`.

---

## 8. Optimizer steps and loss weights

Micro-batches, in the order produced by §7, are grouped into optimizer steps of
`grad_accum_steps` consecutive micro-batches. A trailing group with fewer members forms a
final, shorter step.

Let `u_i` be the number of supervised (mask 1) positions in micro-batch `i`, and let `S`
be the optimizer step containing it. The loss weight of micro-batch `i` is

    loss_weight_i = u_i / sum(u_j for j in S)

Weights within a step sum to 1.

---

## 9. Return value

    {
      "examples": [ {"id": str, "tokens": int64[L], "mask": int64[L]}, ... ],   # L per §6
      "microbatches": [ {"example_ids": [str, ...],
                         "tokens": int64[n, L],
                         "mask": int64[n, L],
                         "loss_weight": float}, ... ],
      "steps": [ {"microbatch_indices": [int, ...], "lr": float}, ... ],
    }

`examples` holds the surviving examples in **input** order. `microbatches` is in packing
order. `steps` is in step order; `microbatch_indices` are indices into `microbatches`.

---

## 10. Learning-rate schedule

Let `T` be the number of optimizer steps and `W = warmup_steps`. For step index
`t` in `[0, T)`:

- if `t < W`:  `lr = base_lr * (t + 1) / W`
- otherwise:   `lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * (t - W) / max(T - W, 1)))`

The schedule advances once per optimizer step, never per micro-batch. When `W = 0` the
warmup branch is unused. When `T <= W` every step is in warmup.
