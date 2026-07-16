import csv, collections, numpy as np

csv_path = r"F:\Astro dev\Astro CS Normalization Database\v33_robustness_test_results.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fail = [r for r in rows if r["success"] == "False" and "Catalog" not in r["fail_reason"]]
ok = [r for r in rows if r["success"] == "True"]

print(f"V3.3 actual algo fails: {len(fail)} frames (from {len(rows)} total, {len(ok)} OK)\n")

for tel in sorted(set(r["telescope"] for r in fail)):
    ft = [r for r in fail if r["telescope"] == tel]
    reasons = collections.Counter()
    for r in ft:
        reasons[r["fail_reason"][:30]] += 1
    stars = [int(r["n_stars"]) for r in ft]
    sat = [int(r["n_saturated"]) for r in ft]
    s0_est = [206.265 * float(r["pixel_size_um"]) / float(r["focal_length_mm"]) for r in ft if float(r["focal_length_mm"]) > 0]
    print(f"{tel} ({len(ft)} frames):")
    for reason, cnt in reasons.most_common():
        print(f"  {reason}: {cnt}")
    print(f"  n_stars: med={np.median(stars):.0f} range=[{min(stars)},{max(stars)}]")
    print(f"  n_sat: med={np.median(sat):.0f} range=[{min(sat)},{max(sat)}]")
    if s0_est:
        print(f"  s0: med={np.median(s0_est):.2f}\"/px range=[{min(s0_est):.2f},{max(s0_est):.2f}]")
    print()

for filt in sorted(set(r["filter_name"] for r in fail)):
    ff = [r for r in fail if r["filter_name"] == filt]
    fok = sum(1 for r in ok if r["filter_name"] == filt)
    print(f"{filt}: {len(ff)} fail / {fok} OK = {fok/(len(ff)+fok)*100:.0f}% succ")

print(f"\n{'='*60}")
print("All fail frames:")
for r in fail:
    print(f"  {r['filename']} [{r['telescope']}/{r['filter_name']}] "
          f"stars={r['n_stars']} sat={r['n_saturated']} s0_est={206.265*float(r['pixel_size_um'])/float(r['focal_length_mm']):.1f}\"/px "
          f"-> {r['fail_reason']}")
