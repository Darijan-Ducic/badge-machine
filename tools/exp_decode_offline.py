import json, sys
sys.path.insert(0, ".")
sys.path.append("tools")
import importlib.util
spec = importlib.util.spec_from_file_location("exp_decode", "tools/exp_decode.py")
ed = importlib.util.module_from_spec(spec); spec.loader.exec_module(ed)
import clips, decode, experiment, runs
from reread import muse_pick

USER, SAVE = "practice:obama-weekly:obama", "--save" in sys.argv
key = json.dumps(decode.CONFIG, sort_keys=True)
cache = {}
for line in open("/tmp/lab-decode-cache.jsonl"):
    e = json.loads(line)
    cache[(e["clip"], e["arm"], e["config"])] = e["item"]
store = clips.backend(None)
recs = [r for r in store.load(USER) if r.get("record") == "v1" and (r.get("label") or {}).get("stt_text")
        and decode.readers(r["samples"])][:40]
recs = [r for r in recs if all((r["id"], a, key) in cache for a in ("refine", "decode"))]
GENERIC = ("we are going to make sure that the american people have the chance to build a better future for our "
           "children and that is why we will keep working together to move this country forward ") * 3
rows = []
for r in recs:
    items = {a: cache[(r["id"], a, key)] for a in ("refine", "decode")}
    stt = r["label"]["stt_text"]
    lo, hi, _ = decode.expected_words(decode.readers(r["samples"]), r["frames"])
    texts = {a: decode.text_of(items[a]) for a in items}
    texts["decode_vis"] = ed.vis_rerank(items["decode"], r["samples"])
    texts["generic@len"] = " ".join(GENERIC.split()[: round((lo + hi) / 2)])
    rows.append({"clip": r["id"], "stt": stt, "control": muse_pick(r["samples"]), "texts": texts, "items": items,
                 "oracle_w": ed.oracle(r["samples"], stt, experiment.word_acc),
                 "oracle_v": ed.oracle(r["samples"], stt, experiment.viseme_acc)})
ed.DERIVED["generic@len"] = "none"
stts = [r["stt"] for r in rows]
gen = [r["texts"]["generic@len"] for r in rows]
print(f"n={len(rows)} obama turns with both arms cached")
spend = 0.0
for arm in ("refine", "decode", "decode_vis", "generic@len"):
    s = ed.summarize(rows, arm)
    spend += s["cost_total"] if arm in ("refine", "decode") else 0
    vs_gen = {}
    for kind in ("word", "viseme"):
        pa, _ = ed.lifts([r["texts"][arm] or "" for r in rows], stts, kind)
        pg, _ = ed.lifts(gen, stts, kind)
        vs_gen[kind] = ed.paired_lift(pg, pa)
    print(f"{arm:<12} LIFT word {s['lift_word_control']:+.3f}>{s['lift_word']:+.3f} ({s['lift_w']['up']}/{s['lift_w']['down']} p={s['lift_w']['p']:.3f}) "
          f"viseme {s['lift_viseme_control']:+.3f}>{s['lift_viseme']:+.3f} ({s['lift_v']['up']}/{s['lift_v']['down']} p={s['lift_v']['p']:.3f}) "
          f"| vs generic@len word {vs_gen['word']['up']}/{vs_gen['word']['down']} p={vs_gen['word']['p']:.3f} "
          f"viseme {vs_gen['viseme']['up']}/{vs_gen['viseme']['down']} p={vs_gen['viseme']['p']:.3f} "
          f"| raw w {s['word_acc_control']:.3f}>{s['word_acc']:.3f} v {s['viseme_acc_control']:.3f}>{s['viseme_acc']:.3f} "
          f"len {s['words_per_stt']:.2f} ${s['extra_cost_per_turn']:.3f} {s['extra_latency_s']:.0f}s")
    if SAVE and arm in ("refine", "decode"):
        rec = ed.run_record(s, rows, user=USER, name="lab-decode-e1", skip=0, hypothesis=ed.HYPOTHESES[arm])
        rec["verdict"] = ("parked: LIFT gain over control is matched by a content-free generic sentence at the "
                          f"duration-predicted length (vs generic@len word {vs_gen['word']['up']}/{vs_gen['word']['down']} "
                          f"p={vs_gen['word']['p']:.2f}, viseme {vs_gen['viseme']['up']}/{vs_gen['viseme']['down']} "
                          f"p={vs_gen['viseme']['p']:.2f}); " + rec["verdict"])
        rec["vs_generic_at_length"] = vs_gen
        runs.save(rec)
        print("  saved", rec["id"])
print(f"spend on these turns ${spend:.2f}")
