import csv, collections, numpy as np

csv_path = r"F:\Astro dev\Astro CS Normalization Database\v33_robustness_test_results.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

ok = [r for r in rows if r["success"] == "True"]
fail = [r for r in rows if r["success"] == "False"]
total = len(rows)
print(f"Total: {total} | OK: {len(ok)} ({len(ok)/total*100:.1f}%) | Fail: {len(fail)}")

tels = collections.Counter(r["telescope"] for r in rows)
tel_ok = collections.Counter(r["telescope"] for r in ok)
for t in sorted(tels):
    print(f"  {t}: {tel_ok[t]}/{tels[t]} ({tel_ok[t]/tels[t]*100:.1f}%)")

if fail:
    for r in fail:
        print(f"  FAIL: {r['filename']} [{r['telescope']}/{r['filter_name']}] -> {r['fail_reason']}")

if ok:
    rms = [float(r["rms_px"]) for r in ok]
    t = [float(r["t_solve_s"]) for r in ok]
    snr = [float(r["theta_snr"]) for r in ok if float(r["theta_snr"]) > 0]
    best_n = [int(r["best_n_range"]) for r in ok]
    corr = [int(r["n_phaseb_corr"]) for r in ok]
    print(f"\nRMS: med={np.median(rms):.3f}px mean={np.mean(rms):.3f}px P25={np.percentile(rms,25):.3f} P75={np.percentile(rms,75):.3f} max={np.max(rms):.3f}")
    print(f"Time: med={np.median(t):.2f}s mean={np.mean(t):.2f}s P25={np.percentile(t,25):.2f} P75={np.percentile(t,75):.2f} max={np.max(t):.2f}")
    print(f"SNR: med={np.median(snr):.0f}x mean={np.mean(snr):.0f}x")
    print(f"best_n_range: med={np.median(best_n):.0f} mean={np.mean(best_n):.0f}")
    print(f"n_corr: med={np.median(corr):.0f} mean={np.mean(corr):.0f}")
