"""Edge feathering thins broad contours without sacrificing fine detail.

The production JavaScript function is executed directly under Node.  The test
uses masks rather than screenshots because its contract is exact: one-pixel
detail stays byte-identical, broad regions converge to connected centre lines,
and the slider is monotone.

Exit code 0 = pass; 1 = fail.
"""
import json
import os
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if not shutil.which("node"):
        print("SKIPPED - node is not on PATH; Edge feather behavior was not tested.")
        return 0

    proc = subprocess.run(
        ["node", os.path.join(HERE, "edge_feather_js.js")],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        print("FAIL - Edge feather harness crashed:")
        print((proc.stderr or proc.stdout)[:2000])
        return 1
    try:
        result = json.loads(proc.stdout)
    except ValueError:
        print("FAIL - Edge feather harness produced no JSON:")
        print(proc.stdout[:1000])
        return 1

    failures = result.get("failures") or []
    if failures:
        print("FAIL (%d)" % len(failures))
        for failure in failures:
            print("  -", failure)
        print("  metrics:", result.get("metrics"))
        return 1

    m = result["metrics"]
    print("ok - one-pixel detail stayed %d/%d pixels; thick bar %d->%d, "
          "thick ring %d->%d; slider mass %s"
          % (m["detailPixelsAt100"], m["detailPixels"],
             m["barPixels"], m["barPixelsAt100"],
             m["ringPixels"], m["ringPixelsAt100"], m["barMass"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
