"""The A/B panel has no live status/help block below its grade rows."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "CNPro_AB.py"
STYLE = ROOT / "style.css"


def main():
    source = SCRIPT.read_text(encoding="utf-8")
    css = STYLE.read_text(encoding="utf-8")
    forbidden = {
        "status component": "cnpro-ab-status",
        "status renderer": "_status_markdown",
        "scale legend": "0 = A much better",
        "empty-search help": "nothing graded yet - the first answer",
    }
    found = [name for name, text in forbidden.items()
             if text in source or text in css]
    if found:
        raise AssertionError("removed A/B block returned: " + ", ".join(found))
    print("ok - the A/B status/help block is absent from the UI and stylesheet")


if __name__ == "__main__":
    main()
