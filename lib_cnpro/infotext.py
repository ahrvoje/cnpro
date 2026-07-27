from typing import List, Tuple, Union

import gradio as gr

from modules.processing import StableDiffusionProcessing

from lib_cnpro import external_code
from lib_cnpro.logging import logger


def field_to_displaytext(fieldname: str) -> str:
    return " ".join([word.capitalize() for word in fieldname.split("_")])


def displaytext_to_field(text: str) -> str:
    return "_".join([word.lower() for word in text.split(" ")])


def parse_value(value: str) -> Union[str, float, int, bool]:
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Plain string.


def serialize_unit(unit: external_code.ControlNetUnit) -> str:
    def include(field, value):
        # Exclude hidden slider values and empty optional profiles; a flat-0.5
        # balance profile (the UI default) is a no-op and would otherwise put a
        # meaningless token into every image's infotext.
        if value in (-1, ""):
            return False
        if field == "balance_profile":
            return not external_code.balance_points_are_neutral(
                external_code.parse_weight_profile(value))
        return True

    log_value = {
        field_to_displaytext(field): getattr(unit, field)
        for field in external_code.ControlNetUnit.infotext_fields()
        if include(field, getattr(unit, field))
    }
    if not all("," not in str(v) and ":" not in str(v) for v in log_value.values()):
        logger.error(f"Unexpected tokens encountered:\n{log_value}")
        return ""

    return ", ".join(f"{field}: {value}" for field, value in log_value.items())


def parse_unit(text: str) -> external_code.ControlNetUnit:
    """Parse one unit's infotext segment, field by field.

    Malformed items are SKIPPED rather than allowed to abort the unit: values
    are written ','/':'-free, but a hand-edited or foreign infotext carrying
    one ': ' too many used to raise here and drop the whole unit's paste - a
    single bad field would silently lose the profile, the model and everything
    else in the same segment.
    """
    values = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition(": ")
        if not sep:
            logger.warning(f"Skipping unparsable ControlNet infotext item: {item!r}")
            continue
        field = displaytext_to_field(key)
        if field not in vars(external_code.ControlNetUnit):
            logger.warning(f"Skipping unknown ControlNet infotext field: {key!r}")
            continue
        values[field] = parse_value(value)
    return external_code.ControlNetUnit(enabled=True, **values)


class Infotext(object):
    def __init__(self) -> None:
        self.infotext_fields: List[Tuple[gr.components.IOComponent, str]] = []
        self.paste_field_names: List[str] = []

    @staticmethod
    def unit_prefix(unit_index: int) -> str:
        return f"ControlNet {unit_index}"

    def register_unit(self, unit_index: int, uigroup) -> None:
        """Register the unit's UI group. By regsitering the unit, A1111 will be
        able to paste values from infotext to IOComponents.

        Args:
            unit_index: The index of the ControlNet unit
            uigroup: The ControlNetUiGroup instance that contains all gradio
                     iocomponents.
        """
        unit_prefix = Infotext.unit_prefix(unit_index)
        for field in external_code.ControlNetUnit.infotext_fields():
            # Every field in ControlNetUnit should have a corresponding
            # IOComponent in ControlNetUiGroup.
            io_component = getattr(uigroup, field)
            component_locator = f"{unit_prefix} {field}"
            self.infotext_fields.append((io_component, component_locator))
            self.paste_field_names.append(component_locator)

    @staticmethod
    def write_infotext(
        units: List[external_code.ControlNetUnit], p: StableDiffusionProcessing
    ):
        """Write infotext to `p`."""
        p.extra_generation_params.update(
            {
                Infotext.unit_prefix(i): serialize_unit(unit)
                for i, unit in enumerate(units)
                if unit.enabled
            }
        )

    @staticmethod
    def on_infotext_pasted(infotext: str, results: dict) -> None:
        """Parse ControlNet infotext string and write result to `results` dict."""
        updates = {}
        for k, v in results.items():
            if not k.startswith("ControlNet"):
                continue

            assert isinstance(v, str), f"Expect string but got {v}."
            try:
                unit = parse_unit(v)
                if not unit.weight_profile:
                    # Legacy infotext: convert constant weight + timestep range
                    # into an equivalent weight profile.
                    unit.weight_profile = external_code.weight_profile_from_scalars(
                        unit.weight, unit.guidance_start, unit.guidance_end
                    )
                if not unit.balance_profile:
                    # Legacy infotext: approximate the old Control Mode chooser
                    # with a flat balance profile. BALANCED (and any new-format
                    # infotext, which omits neutral balance on write) pastes
                    # the EXPLICIT neutral string, never "" - an empty textbox
                    # cannot be drawn by the editor, which would keep showing
                    # its previous curve while the unit held neutral (and the
                    # next edit would resurrect the stale curve).
                    mode = unit.control_mode
                    mode = getattr(mode, "value", mode)
                    unit.balance_profile = {
                        external_code.ControlMode.CONTROL.value: "0@1;1@1",
                        external_code.ControlMode.PROMPT.value: "0@0.25;1@0.25",
                    }.get(mode, "0@0.5;1@0.5")
                for field, value in vars(unit).items():
                    if field == "image":
                        continue
                    if value is None:
                        logger.debug(f"InfoText: Skipping {field} because value is None.")
                        continue

                    component_locator = f"{k} {field}"
                    updates[component_locator] = value
                    logger.debug(f"InfoText: Setting {component_locator} = {value}")
            except Exception as e:
                logger.warn(
                    f"Failed to parse infotext, legacy format infotext is no longer supported:\n{v}\n{e}"
                )

        results.update(updates)
