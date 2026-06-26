import os
BASE = r"F:\nano-everything\mini-control-engineering-practice\14. mini-soft-sensor-inferential\mini-soft-sensor-maintenance-aging"
def writef(relpath, content):
    path = os.path.join(BASE, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {relpath}: {content.count(chr(10))} lines")
