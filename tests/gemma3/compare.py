# """gemma3 layer dump comparison — reads the two dumps and says where they part.

#     python tests/gemma3/compare.py [--vai DIR] [--torch DIR] [--tol 0.02]

# Columns, per tap:
#   shape        the PyTorch shape, after leading 1s are squeezed
#   max|d|       largest absolute difference
#   rms          root-mean-square difference
#   rel          rms(diff) / rms(torch) — the scale-free one, and what the
#                growth column is computed from
#   cos          cosine similarity over the flattened tensor
#   growth       rel of this tap / rel of the previous one in the same stack:
#                ~1 means the layer passed the error through, >1 means it made
#                its own, <1 means a norm pulled it back

# Logits get their own section: a difference only matters where it changes the
# argmax, so the rows carry both sides' top-1 and the margin that decided it.
# """

# import argparse
# import os
# import sys
# import numpy as np


# def load(path):
#     """As [rows, D]: a batch axis is dropped and a bare vector becomes one row,
#     so a sliced tap ([1, D]) and a full one ([T, D]) still line up."""
#     a = np.load(path).astype(np.float64)
#     while a.ndim > 2 and a.shape[0] == 1:      # [1, T, D] -> [T, D]
#         a = a[0]
#     if a.ndim == 1:
#         a = a[None, :]
#     if a.ndim > 2:                             # anything else: flatten to rows
#         a = a.reshape(-1, a.shape[-1])
#     return a


# def stats(a, b):
#     if a.shape[-1] != b.shape[-1]:
#         return None
#     # vAi slices the last decoder layer to the final row, and the lm head reads
#     # that row only. Whichever side carries fewer rows decides the overlap, and
#     # it is always the tail of the sequence.
#     n = min(a.shape[0], b.shape[0])
#     a, b = a[-n:], b[-n:]

#     d = np.abs(a - b)
#     rms_b = np.sqrt((b ** 2).mean())
#     rel = np.sqrt((d ** 2).mean()) / rms_b if rms_b > 0 else float("nan")
#     af, bf = a.ravel(), b.ravel()
#     denom = np.linalg.norm(af) * np.linalg.norm(bf)
#     cos = float(af @ bf / denom) if denom > 0 else float("nan")
#     return dict(shape=tuple(a.shape), mx=d.max(), rms=np.sqrt((d ** 2).mean()),
#                 rel=rel, cos=cos, ref=rms_b, refmax=np.abs(b).max())


# SUMMARY = {}


# def section(title, names, vai_dir, torch_dir, tol):
#     rows, prev = [], None
#     for name in names:
#         pv = os.path.join(vai_dir, name + ".npy")
#         pt = os.path.join(torch_dir, name + ".npy")
#         if not (os.path.exists(pv) and os.path.exists(pt)):
#             continue
#         s = stats(load(pv), load(pt))
#         if s is None:
#             rows.append((name, "shape mismatch", None))
#             continue
#         growth = (s["rel"] / prev) if (prev and prev > 0) else None
#         prev = s["rel"]
#         rows.append((name, s, growth))
#         SUMMARY.setdefault(title, []).append((name, s["rel"]))

#     if not rows:
#         return
#     print(f"\n{title}")
#     print(f"  {'tap':<16} {'shape':>13} {'|ref|rms':>10} {'|ref|max':>10} "
#           f"{'max|d|':>10} {'rel':>9} {'cos':>9} {'growth':>7}")
#     print("  " + "-" * 94)
#     for name, s, growth in rows:
#         if not isinstance(s, dict):
#             print(f"  {name:<16} {s}")
#             continue
#         flag = "  <--" if s["rel"] > tol else ""
#         g = f"{growth:7.2f}" if growth else "      ."
#         print(f"  {name:<16} {str(s['shape']):>13} {s['ref']:10.3f} {s['refmax']:10.1f} "
#               f"{s['mx']:10.4f} {s['rel']:9.2e} {s['cos']:9.6f} {g}{flag}")


# def logit_row(name, vai_dir, torch_dir, vocab_names=None):
#     pv, pt = os.path.join(vai_dir, name + ".npy"), os.path.join(torch_dir, name + ".npy")
#     if not (os.path.exists(pv) and os.path.exists(pt)):
#         return None
#     # The lm head is read at the last position only; torch keeps every row.
#     a, b = load(pv)[-1], load(pt)[-1]
#     ia, ib = int(a.argmax()), int(b.argmax())
#     order = np.argsort(-b)
#     rank_of_vai = int(np.where(order == ia)[0][0])
#     top2_gap = float(b[order[0]] - b[order[1]])
#     return dict(name=name, same=ia == ib, vai=ia, torch=ib,
#                 rank=rank_of_vai, gap=top2_gap,
#                 mx=float(np.abs(a - b).max()),
#                 rel=float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean())))


# def main():
#     ap = argparse.ArgumentParser()
#     root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     ap.add_argument("--vai", default=os.path.join(root, "tests/gemma3/dumps/vai-fp32"))
#     ap.add_argument("--torch", default=os.path.join(root, "tests/gemma3/dumps/torch"))
#     ap.add_argument("--tol", type=float, default=0.02)
#     ap.add_argument("--tokenizer", default=os.path.join(root, "data/gemma3/4B"))
#     args = ap.parse_args()

#     for d in (args.vai, args.torch):
#         if not os.path.isdir(d):
#             sys.exit(f"missing dump directory: {d}\n"
#                      f"run  python tests/gemma3/dump_torch.py  then  build/bin/test-gemma3-dump")

#     print(f"left  (measured)  : {args.vai}")
#     print(f"right (reference) : {args.torch}")

#     # The prompts have to be identical before anything downstream means much.
#     pv = os.path.join(args.vai, "prompt.ids.npy")
#     pt = os.path.join(args.torch, "prompt.ids.npy")
#     if os.path.exists(pv) and os.path.exists(pt):
#         a, b = np.load(pv).ravel(), np.load(pt).ravel()
#         same = a.size == b.size and bool((a == b).all())
#         print(f"prompt ids : {a.size} vs {b.size} tokens — "
#               + ("identical" if same else "DIFFERENT"))
#         if not same:
#             n = min(a.size, b.size)
#             first = int(np.argmax(a[:n] != b[:n]))
#             print(f"             first difference at {first}: vAi {a[first]} vs torch {b[first]}")

#     section("vision tower", ["vis.embed"] + [f"vis.{i:02d}" for i in range(27)]
#             + ["vis.post_norm", "vis.pool", "vis.pool_norm", "vis.proj"],
#             args.vai, args.torch, args.tol)

#     section("text decoder (image prefill)",
#             ["txt.embed"] + [f"txt.{i:02d}" for i in range(34)] + ["txt.final_norm"],
#             args.vai, args.torch, args.tol)

#     section("text decoder (decode step 0)",
#             ["step00.txt.embed_text"] + [f"step00.txt.{i:02d}" for i in range(34)]
#             + ["step00.txt.final_norm"],
#             args.vai, args.torch, args.tol)

#     # ----- logits ----------------------------------------------------------
#     decode = None
#     try:
#         from transformers import AutoTokenizer
#         decode = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
#     except Exception:
#         pass

#     rows = [r for r in [logit_row("logits", args.vai, args.torch)] if r]
#     step = 0
#     while True:
#         r = logit_row(f"step.{step:02d}.logits", args.vai, args.torch)
#         if r is None:
#             break
#         rows.append(r)
#         step += 1

#     if rows:
#         print("\nlogits — a difference only counts where it moves the argmax")
#         print(f"  {'tap':<18} {'rel':>9} {'max|d|':>9}  {'top1':<22} "
#               f"{'vAi rank':>8} {'torch top1-top2':>16}")
#         print("  " + "-" * 90)
#         for r in rows:
#             def piece(t):
#                 if decode is None:
#                     return str(t)
#                 return f"{t} {decode.decode([t])!r}"
#             mark = "same" if r["same"] else "DIFF"
#             top = piece(r["torch"]) if r["same"] else f"{piece(r['vai'])} vs {piece(r['torch'])}"
#             print(f"  {r['name']:<18} {r['rel']:9.2e} {r['mx']:9.4f}  {mark} {top:<22} "
#                   f"{r['rank']:8d} {r['gap']:16.4f}")

#     # ----- what each side picked ------------------------------------------
#     pv = os.path.join(args.vai, "steps.picks.npy")
#     pt = os.path.join(args.torch, "steps.picks.npy")
#     fv = os.path.join(args.vai, "steps.fed.npy")
#     ft = os.path.join(args.torch, "steps.ids.npy")
#     if all(os.path.exists(p) for p in (pv, pt, fv, ft)):
#         a, b = np.load(pv).ravel(), np.load(pt).ravel()
#         fa, fb = np.load(fv).ravel(), np.load(ft).ravel()
#         n = min(a.size, b.size)
#         forced = fa.size == fb.size and bool((fa == fb).all())
#         print(f"\ndecode steps — inputs {'teacher-forced (identical)' if forced else 'NOT identical'}")
#         print(f"  {'step':>4}  {'fed':<22} {'vAi pick':<22} {'torch pick':<22}")
#         print("  " + "-" * 74)
#         for i in range(n):
#             def piece(t):
#                 return f"{t} {decode.decode([int(t)])!r}" if decode else str(t)
#             flag = "" if a[i] == b[i] else "   <-- differs"
#             print(f"  {i:4d}  {piece(fa[i]):<22} {piece(a[i]):<22} {piece(b[i]):<22}{flag}")
#         agree = int(np.argmax(a[:n] != b[:n])) if (a[:n] != b[:n]).any() else n
#         print(f"\n  first differing pick: step {agree}" if agree < n
#               else f"\n  every pick agrees ({n} steps)")

#     # ----- summary ---------------------------------------------------------
#     if SUMMARY:
#         print("\nsummary — relative error entering and leaving each stack")
#         print(f"  {'stack':<32} {'first tap':>14} {'last tap':>14} {'end/start':>10}")
#         print("  " + "-" * 74)
#         for title, entries in SUMMARY.items():
#             first, last = entries[0], entries[-1]
#             ratio = last[1] / first[1] if first[1] > 0 else float("nan")
#             print(f"  {title:<32} {first[1]:14.2e} {last[1]:14.2e} {ratio:10.1f}")
#         print("\n  A stack that starts low and ends low did its own arithmetic well;")
#         print("  one that starts high inherited the error from the stack before it.")


# if __name__ == "__main__":
#     main()

"""gemma3 layer dump comparison — reads the two dumps and says where they part.

    python tests/gemma3/compare.py [--vai DIR] [--torch DIR] [--tol 0.02]

Columns, per tap:
  shape        the PyTorch shape, after leading 1s are squeezed
  |ref|rms     RMS magnitude of the PyTorch reference tensor
  |ref|max     maximum absolute value of the PyTorch reference tensor
  mean|d|      mean absolute difference (MAE)
  rms          root-mean-square difference (RMSE)
  max|d|       largest absolute difference
  rel          rms(diff) / rms(torch) — the scale-free one, and what the
               growth column is computed from
  cos          cosine similarity over the flattened tensor
  growth       rel of this tap / rel of the previous one in the same stack:
               ~1 means the layer passed the error through,
               >1 means it made/amplified error,
               <1 means a norm or averaging operation pulled it back

Logits get their own section: a difference only matters where it changes the
argmax, so the rows carry both sides' top-1 and the margin that decided it.
"""

import argparse
import os
import sys
import numpy as np


def load(path):
    """Load as [rows, D].

    A batch axis is dropped and a bare vector becomes one row, so a sliced tap
    ([1, D]) and a full one ([T, D]) still line up.
    """
    a = np.load(path).astype(np.float64)

    while a.ndim > 2 and a.shape[0] == 1:
        # [1, T, D] -> [T, D]
        a = a[0]

    if a.ndim == 1:
        a = a[None, :]

    if a.ndim > 2:
        # anything else: flatten to rows
        a = a.reshape(-1, a.shape[-1])

    return a


def stats(a, b):
    """Compute numerical difference statistics.

    a = vAi measured result
    b = PyTorch reference result
    """
    if a.shape[-1] != b.shape[-1]:
        return None

    # vAi slices the last decoder layer to the final row, and the lm head reads
    # that row only. Whichever side carries fewer rows decides the overlap, and
    # it is always the tail of the sequence.
    n = min(a.shape[0], b.shape[0])
    a, b = a[-n:], b[-n:]

    diff = a - b
    abs_diff = np.abs(diff)

    # Reference magnitude
    rms_b = np.sqrt((b ** 2).mean())
    ref_max = np.abs(b).max()

    # Error metrics
    mae = abs_diff.mean()
    rmse = np.sqrt((diff ** 2).mean())
    max_abs = abs_diff.max()

    # Scale-free relative RMS error
    rel = rmse / rms_b if rms_b > 0 else float("nan")

    # Cosine similarity
    af = a.ravel()
    bf = b.ravel()

    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    cos = float(af @ bf / denom) if denom > 0 else float("nan")

    return dict(
        shape=tuple(a.shape),
        ref=rms_b,
        refmax=ref_max,
        mae=mae,
        rms=rmse,
        mx=max_abs,
        rel=rel,
        cos=cos,
    )


SUMMARY = {}


def section(title, names, vai_dir, torch_dir, tol):
    rows = []
    prev = None

    for name in names:
        pv = os.path.join(vai_dir, name + ".npy")
        pt = os.path.join(torch_dir, name + ".npy")

        if not (os.path.exists(pv) and os.path.exists(pt)):
            continue

        s = stats(load(pv), load(pt))

        if s is None:
            rows.append((name, "shape mismatch", None))
            continue

        growth = (s["rel"] / prev) if (prev is not None and prev > 0) else None
        prev = s["rel"]

        rows.append((name, s, growth))
        SUMMARY.setdefault(title, []).append((name, s["rel"]))

    if not rows:
        return

    print(f"\n{title}")

    print(
        f"  {'tap':<16} "
        f"{'shape':>13} "
        f"{'|ref|rms':>10} "
        f"{'|ref|max':>10} "
        f"{'mean|d|':>10} "
        f"{'rms':>10} "
        f"{'max|d|':>10} "
        f"{'rel':>9} "
        f"{'cos':>9} "
        f"{'growth':>7}"
    )

    print("  " + "-" * 118)

    for name, s, growth in rows:
        if not isinstance(s, dict):
            print(f"  {name:<16} {s}")
            continue

        flag = "  <--" if s["rel"] > tol else ""

        if growth is not None:
            g = f"{growth:7.2f}"
        else:
            g = "      ."

        print(
            f"  {name:<16} "
            f"{str(s['shape']):>13} "
            f"{s['ref']:10.3f} "
            f"{s['refmax']:10.1f} "
            f"{s['mae']:10.6f} "
            f"{s['rms']:10.6f} "
            f"{s['mx']:10.4f} "
            f"{s['rel']:9.2e} "
            f"{s['cos']:9.6f} "
            f"{g}{flag}"
        )


def logit_row(name, vai_dir, torch_dir, vocab_names=None):
    pv = os.path.join(vai_dir, name + ".npy")
    pt = os.path.join(torch_dir, name + ".npy")

    if not (os.path.exists(pv) and os.path.exists(pt)):
        return None

    # The lm head is read at the last position only;
    # torch keeps every row.
    a = load(pv)[-1]
    b = load(pt)[-1]

    diff = a - b
    abs_diff = np.abs(diff)

    ia = int(a.argmax())
    ib = int(b.argmax())

    order = np.argsort(-b)
    rank_of_vai = int(np.where(order == ia)[0][0])
    top2_gap = float(b[order[0]] - b[order[1]])

    rms_b = np.sqrt((b ** 2).mean())
    rmse = np.sqrt((diff ** 2).mean())

    return dict(
        name=name,
        same=ia == ib,
        vai=ia,
        torch=ib,
        rank=rank_of_vai,
        gap=top2_gap,
        mae=float(abs_diff.mean()),
        rms=float(rmse),
        mx=float(abs_diff.max()),
        rel=float(rmse / rms_b) if rms_b > 0 else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()

    root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    ap.add_argument(
        "--vai",
        # default=os.path.join(root, "tests/gemma3/dumps/vai")
        default=os.path.join(root, "tests/gemma3/dumps/torch_bf16")
    )

    ap.add_argument(
        "--torch",
        default=os.path.join(root, "tests/gemma3/dumps/torch_32")
        # default=os.path.join(root, "tests/gemma3/dumps/torch_bf16")
    )

    ap.add_argument(
        "--tol",
        type=float,
        default=0.02
    )

    ap.add_argument(
        "--tokenizer",
        default=os.path.join(root, "data/gemma3/4B")
    )

    args = ap.parse_args()

    for d in (args.vai, args.torch):
        if not os.path.isdir(d):
            sys.exit(
                f"missing dump directory: {d}\n"
                f"run  python tests/gemma3/dump_torch.py  "
                f"then  build/bin/test-gemma3-dump"
            )

    print(f"left  (measured)  : {args.vai}")
    print(f"right (reference) : {args.torch}")

    # ----------------------------------------------------------------------
    # Prompt IDs
    # ----------------------------------------------------------------------

    # The prompts have to be identical before anything downstream means much.
    pv = os.path.join(args.vai, "prompt.ids.npy")
    pt = os.path.join(args.torch, "prompt.ids.npy")

    if os.path.exists(pv) and os.path.exists(pt):
        a = np.load(pv).ravel()
        b = np.load(pt).ravel()

        same = a.size == b.size and bool((a == b).all())

        print(
            f"prompt ids : {a.size} vs {b.size} tokens — "
            + ("identical" if same else "DIFFERENT")
        )

        if not same:
            n = min(a.size, b.size)

            mismatch = a[:n] != b[:n]

            if mismatch.any():
                first = int(np.argmax(mismatch))

                print(
                    f"             first difference at {first}: "
                    f"vAi {a[first]} vs torch {b[first]}"
                )
            elif a.size != b.size:
                print(
                    f"             token count differs: "
                    f"vAi {a.size}, torch {b.size}"
                )

    # ----------------------------------------------------------------------
    # Vision tower
    # ----------------------------------------------------------------------

    section(
        "vision tower",
        ["vis.embed"]
        + [f"vis.{i:02d}" for i in range(27)]
        + [
            "vis.post_norm",
            "vis.pool",
            "vis.pool_norm",
            "vis.proj",
        ],
        args.vai,
        args.torch,
        args.tol,
    )

    # ----------------------------------------------------------------------
    # Text decoder: image prefill
    # ----------------------------------------------------------------------

    section(
        "text decoder (image prefill)",
        ["txt.embed"]
        + [f"txt.{i:02d}" for i in range(34)]
        + ["txt.final_norm"],
        args.vai,
        args.torch,
        args.tol,
    )

    # ----------------------------------------------------------------------
    # Text decoder: decode step 0
    # ----------------------------------------------------------------------

    section(
        "text decoder (decode step 0)",
        ["step00.txt.embed_text"]
        + [f"step00.txt.{i:02d}" for i in range(34)]
        + ["step00.txt.final_norm"],
        args.vai,
        args.torch,
        args.tol,
    )

    # ----------------------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------------------

    decode = None

    try:
        from transformers import AutoTokenizer

        decode = AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=True,
        )

    except Exception:
        pass

    # ----------------------------------------------------------------------
    # Logits
    # ----------------------------------------------------------------------

    rows = [
        r
        for r in [
            logit_row(
                "logits",
                args.vai,
                args.torch,
            )
        ]
        if r
    ]

    step = 0

    while True:
        r = logit_row(
            f"step.{step:02d}.logits",
            args.vai,
            args.torch,
        )

        if r is None:
            break

        rows.append(r)
        step += 1

    if rows:
        print(
            "\nlogits — numerical error plus whether it moves the argmax"
        )

        print(
            f"  {'tap':<18} "
            f"{'mean|d|':>10} "
            f"{'rms':>10} "
            f"{'max|d|':>9} "
            f"{'rel':>9}  "
            f"{'top1':<22} "
            f"{'vAi rank':>8} "
            f"{'torch top1-top2':>16}"
        )

        print("  " + "-" * 122)

        for r in rows:

            def piece(t):
                if decode is None:
                    return str(t)

                return f"{t} {decode.decode([t])!r}"

            mark = "same" if r["same"] else "DIFF"

            if r["same"]:
                top = piece(r["torch"])
            else:
                top = (
                    f"{piece(r['vai'])} "
                    f"vs {piece(r['torch'])}"
                )

            print(
                f"  {r['name']:<18} "
                f"{r['mae']:10.6f} "
                f"{r['rms']:10.6f} "
                f"{r['mx']:9.4f} "
                f"{r['rel']:9.2e}  "
                f"{mark} {top:<22} "
                f"{r['rank']:8d} "
                f"{r['gap']:16.4f}"
            )

    # ----------------------------------------------------------------------
    # What each side picked
    # ----------------------------------------------------------------------

    pv = os.path.join(args.vai, "steps.picks.npy")
    pt = os.path.join(args.torch, "steps.picks.npy")

    fv = os.path.join(args.vai, "steps.fed.npy")
    ft = os.path.join(args.torch, "steps.ids.npy")

    if all(
        os.path.exists(p)
        for p in (pv, pt, fv, ft)
    ):
        a = np.load(pv).ravel()
        b = np.load(pt).ravel()

        fa = np.load(fv).ravel()
        fb = np.load(ft).ravel()

        n = min(a.size, b.size)

        forced = (
            fa.size == fb.size
            and bool((fa == fb).all())
        )

        print(
            f"\ndecode steps — inputs "
            f"{'teacher-forced (identical)' if forced else 'NOT identical'}"
        )

        print(
            f"  {'step':>4}  "
            f"{'fed':<22} "
            f"{'vAi pick':<22} "
            f"{'torch pick':<22}"
        )

        print("  " + "-" * 74)

        for i in range(n):

            def piece(t):
                if decode:
                    return (
                        f"{t} "
                        f"{decode.decode([int(t)])!r}"
                    )

                return str(t)

            flag = (
                ""
                if a[i] == b[i]
                else "   <-- differs"
            )

            print(
                f"  {i:4d}  "
                f"{piece(fa[i]):<22} "
                f"{piece(a[i]):<22} "
                f"{piece(b[i]):<22}"
                f"{flag}"
            )

        mismatch = a[:n] != b[:n]

        if mismatch.any():
            agree = int(np.argmax(mismatch))

            print(
                f"\n  first differing pick: step {agree}"
            )

        else:
            print(
                f"\n  every pick agrees ({n} steps)"
            )

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------

    if SUMMARY:
        print(
            "\nsummary — relative error entering "
            "and leaving each stack"
        )

        print(
            f"  {'stack':<32} "
            f"{'first tap':>14} "
            f"{'last tap':>14} "
            f"{'end/start':>10}"
        )

        print("  " + "-" * 74)

        for title, entries in SUMMARY.items():
            first = entries[0]
            last = entries[-1]

            ratio = (
                last[1] / first[1]
                if first[1] > 0
                else float("nan")
            )

            print(
                f"  {title:<32} "
                f"{first[1]:14.2e} "
                f"{last[1]:14.2e} "
                f"{ratio:10.1f}"
            )

        print(
            "\n"
            "  Error metric interpretation:\n"
            "    mean|d| : average absolute element-wise error (MAE)\n"
            "    rms     : root-mean-square element-wise error (RMSE)\n"
            "    max|d|  : worst single element error\n"
            "    rel     : RMSE / reference RMS\n"
            "    growth  : current rel / previous rel\n"
        )

        print(
            "  A stack that starts low and ends low "
            "did its own arithmetic well;"
        )

        print(
            "  one that starts high inherited the "
            "error from the stack before it."
        )


if __name__ == "__main__":
    main()