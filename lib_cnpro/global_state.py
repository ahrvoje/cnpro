import json
import os.path
import stat
import struct
from collections import OrderedDict

from modules import shared, sd_models
# --- host boundary: resolved through cnpro_host/adapter.py (never import the host directly) ---
from cnpro_host.adapter import model_dir, preprocessors

controlnet_dir = model_dir()
# the SAME dict object the host mutates as preprocessors register -- binding
# it once is intentional and matches the host's own usage
supported_preprocessors = preprocessors()

from typing import Tuple, List

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".patch"]

#: CNPro's preferred preprocessor resolution, for every preprocessor and every
#: model family. The host's `Preprocessor` base declares 512, which is an SD1.5-era
#: number: SDXL, Flux and Z-Image all generate at 1024+, and preprocessing a 1024
#: image at 512 throws away detail the control model could have used, before it
#: ever sees it.
#:
#: One constant rather than a per-family table on purpose. "Which resolution
#: suits this architecture" is a question about the IMAGE being preprocessed, not
#: about the control model - canny at 1024 is canny at 1024 whether a UNet or a
#: DiT consumes it - and a per-family table would be one more declared-here /
#: honoured-there pair to drift (ARCHITECTURE.md section 8).
DEFAULT_PROCESSOR_RES = 1024


def default_processor_res(preprocessor) -> int:
    """CNPro's preferred resolution, clamped into `preprocessor`'s own range.

    Returns the preprocessor's OWN value untouched when its resolution slider is
    hidden. A hidden slider means the preprocessor does not use a resolution at
    all (inpaint, reference, tile), and those declare
    ``PreprocessorParameter(visible=False)`` -- which resets the range to the
    base 0.0..1.0, so forcing 1024 there would clamp to 1 and put a meaningless
    number in the infotext of every such unit.

    Otherwise 1024 is clamped to the declared [minimum, maximum] and snapped to
    `step`, because a slider value outside its own bounds is a gradio warning at
    best and a silently-rejected value at worst.
    """
    kwargs = getattr(preprocessor, "slider_resolution", None)
    kwargs = getattr(kwargs, "gradio_update_kwargs", None) or {}

    if not kwargs.get("visible", True):
        return int(kwargs.get("value", DEFAULT_PROCESSOR_RES))

    lo = kwargs.get("minimum", 64)
    hi = kwargs.get("maximum", 2048)
    step = kwargs.get("step", 1) or 1

    value = min(max(DEFAULT_PROCESSOR_RES, lo), hi)
    # snap to the slider's grid, measuring from `lo` (gradio's own origin), then
    # re-clamp: rounding can push the last step past `hi`.
    value = lo + round((value - lo) / step) * step
    return int(min(max(value, lo), hi))


def traverse_all_files(curr_path, model_list):
    f_list = [
        (os.path.join(curr_path, entry.name), entry.stat())
        for entry in os.scandir(curr_path)
        if os.path.isdir(curr_path)
    ]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in CN_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list


def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f" [{sd_models.model_hash(filename)}]"] = filename

    return res


controlnet_filename_dict = {'None': 'model.safetensors'}
controlnet_names = ['None']


def get_preprocessor(name):
    return supported_preprocessors.get(name, None)

def get_default_preprocessor(tag):
    ps = get_filtered_preprocessor_names(tag)
    assert len(ps) > 0
    return ps[0] if len(ps) == 1 else ps[1]

def get_sorted_preprocessors():
    preprocessors = [p for k, p in supported_preprocessors.items() if k != 'None']
    preprocessors = sorted(preprocessors, key=lambda x: str(x.sorting_priority).zfill(8) + x.name)[::-1]
    results = OrderedDict()
    results['None'] = supported_preprocessors['None']
    for p in preprocessors:
        results[p.name] = p
    return results


def get_all_controlnet_names():
    return controlnet_names


def get_controlnet_filename(controlnet_name):
    return controlnet_filename_dict[controlnet_name]


# ---- cheap model-type classification (UI gating, e.g. the per-unit prompt) --
#
# The patcher type is only truly known after loading, but the KIND of file is
# readable from the safetensors HEADER alone (the key names - a few KB of
# JSON, no tensor data). The key signatures below mirror the loaders' own
# checks (ControlNetPatcher / ControlLLLitePatcher / IPAdapterPatcher
# try_build_from_state_dict), so UI gating and load reality cannot drift far.

controlnet_type_cache = {}


def _classify_state_dict_keys(keys):
    if any('lllite' in k for k in keys):
        return 'lllite'
    if 'lora_controlnet' in keys:
        return 'controllora'
    if 'controlnet_cond_embedding.conv_in.weight' in keys \
            or 'input_hint_block.0.weight' in keys \
            or any(k.startswith('control_model.') for k in keys):
        return 'controlnet'
    if any(k.startswith(('ip_adapter.', 'image_proj.')) for k in keys):
        return 'ipadapter'
    if any(k.startswith(('adapter.', 'body.')) for k in keys) or 'style_embedding' in keys:
        return 't2i'
    return 'unknown'


def classify_controlnet_type(controlnet_name):
    """Model kind of a dropdown entry: 'controlnet' / 'controllora' /
    'lllite' / 'ipadapter' / 't2i' / 'none' / 'unknown'.

    safetensors files are classified from their header keys; legacy pickle
    formats (.pth/.bin/...) fall back to filename heuristics and otherwise
    report 'unknown' (which callers should fail OPEN on - classic SD15 .pth
    ControlNets land here). Cached per (path, mtime).
    """
    if controlnet_name in (None, '', 'None'):
        return 'none'
    filename = controlnet_filename_dict.get(controlnet_name)
    if not filename:
        return 'unknown'
    try:
        key = (filename, os.path.getmtime(filename))
    except OSError:
        return 'unknown'
    cached = controlnet_type_cache.get(key)
    if cached:
        return cached
    kind = 'unknown'
    lower = os.path.basename(filename).lower()
    if filename.lower().endswith('.safetensors'):
        try:
            with open(filename, 'rb') as file:
                header_size = struct.unpack('<Q', file.read(8))[0]
                header = json.loads(file.read(min(header_size, 1 << 27)))
            kind = _classify_state_dict_keys(list(header.keys()))
        except Exception:
            kind = 'unknown'
    elif 'lllite' in lower:
        kind = 'lllite'
    elif 'ip-adapter' in lower or 'ip_adapter' in lower or 'instantid' in lower or 'instant_id' in lower:
        kind = 'ipadapter'
    elif 't2i' in lower:
        kind = 't2i'
    controlnet_type_cache[key] = kind
    return kind


def get_all_preprocessor_names():
    return list(get_sorted_preprocessors().keys())


def get_all_preprocessor_tags():
    tags = []
    for k, p in supported_preprocessors.items():
        tags += p.tags
    tags = list(set(tags))
    tags = sorted(tags)
    return ['All'] + tags


def get_filtered_preprocessors(tag):
    if tag == 'All':
        return supported_preprocessors
    return {k: v for k, v in get_sorted_preprocessors().items() if tag in v.tags or k == 'None'}


def get_filtered_preprocessor_names(tag):
    return list(get_filtered_preprocessors(tag).keys())


def get_filtered_controlnet_names(tag):
    filtered_preprocessors = get_filtered_preprocessors(tag)
    model_filename_filters = []
    for p in filtered_preprocessors.values():
        model_filename_filters += p.model_filename_filters
    return [x for x in controlnet_names if x == 'None' or any(f.lower() in x.lower() for f in model_filename_filters)]


def update_controlnet_filenames():
    global controlnet_filename_dict, controlnet_names

    controlnet_filename_dict = {'None': 'model.safetensors'}
    controlnet_names = ['None']

    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                        if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [controlnet_dir, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get("control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        controlnet_filename_dict.update(found)

    controlnet_names = list(controlnet_filename_dict.keys())
    return


def select_control_type(
    control_type: str,
) -> Tuple[List[str], List[str], str, str]:
    global controlnet_names

    pattern = control_type.lower()
    all_models = list(controlnet_names)

    if pattern == "all":
        preprocessors = get_sorted_preprocessors().values()
        return [
            [p.name for p in preprocessors],
            all_models,
            'none',  # default option
            "None"   # default model
        ]

    filtered_model_list = get_filtered_controlnet_names(control_type)

    if pattern == "none":
        filtered_model_list.append("None")

    assert len(filtered_model_list) > 0, "'None' model should always be available."
    if len(filtered_model_list) == 1:
        default_model = "None"
    else:
        default_model = filtered_model_list[1]
        for x in filtered_model_list:
            if "11" in x.split("[")[0]:
                default_model = x
                break

    return (
        get_filtered_preprocessor_names(control_type),
        filtered_model_list,
        get_default_preprocessor(control_type),
        default_model
    )
