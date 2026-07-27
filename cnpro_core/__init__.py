"""L1 -- CNPro's host-agnostic core.

HARD RULE: nothing in this package may import the host (no `backend.*`, no
`modules*`, no gradio) or CNPro's own L2/L3. It is plain python over floats and
lists, and that is what makes it portable to ComfyUI or any future host
unchanged. The parity test runs it under a bare interpreter to keep that honest.
"""
