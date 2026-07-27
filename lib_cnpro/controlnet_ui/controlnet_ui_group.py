import copy
import json
import dataclasses
from dataclasses import dataclass
import gradio as gr
from typing import List, Optional
import numpy as np

from lib_cnpro.utils import judge_image_type
from lib_cnpro import (
    global_state,
    external_code,
)
from lib_cnpro.logging import logger
from lib_cnpro.controlnet_ui.openpose_editor import OpenposeEditor
from lib_cnpro.controlnet_ui.photopea import Photopea
from lib_cnpro.enums import HiResFixOption
from modules import shared
# --- host boundary: resolved through cnpro_host/adapter.py (never import the host directly) ---
from cnpro_host.adapter import image_utils

HWC3, _to_torch = image_utils()
from lib_cnpro.external_code import UiControlNetUnit
from modules.ui_components import ToolButton
from cnpro_host.adapter import canvas_widget

ForgeCanvas, LogicalImage = canvas_widget()


@dataclass
class A1111Context:
    """Contains all components from A1111."""

    img2img_batch_input_dir = None
    img2img_batch_output_dir = None
    txt2img_submit_button = None
    img2img_submit_button = None

    # Slider controls from A1111 WebUI.
    txt2img_w_slider = None
    txt2img_h_slider = None
    img2img_w_slider = None
    img2img_h_slider = None

    img2img_img2img_tab = None
    img2img_img2img_sketch_tab = None
    img2img_batch_tab = None
    img2img_inpaint_tab = None
    img2img_inpaint_sketch_tab = None
    img2img_inpaint_upload_tab = None

    img2img_inpaint_area = None
    txt2img_enable_hr = None

    @property
    def img2img_inpaint_tabs(self):
        return (
            self.img2img_inpaint_tab,
            self.img2img_inpaint_sketch_tab,
            self.img2img_inpaint_upload_tab,
        )

    @property
    def img2img_non_inpaint_tabs(self):
        return (
            self.img2img_img2img_tab,
            self.img2img_img2img_sketch_tab,
            self.img2img_batch_tab,
        )

    @property
    def ui_initialized(self) -> bool:
        optional_components = {
            # Optional components are only available after A1111 v1.7.0.
            "img2img_img2img_tab": "img2img_img2img_tab",
            "img2img_img2img_sketch_tab": "img2img_img2img_sketch_tab",
            "img2img_batch_tab": "img2img_batch_tab",
            "img2img_inpaint_tab": "img2img_inpaint_tab",
            "img2img_inpaint_sketch_tab": "img2img_inpaint_sketch_tab",
            "img2img_inpaint_upload_tab": "img2img_inpaint_upload_tab",
        }
        return all(
            c
            for name, c in vars(self).items()
            if name not in optional_components.values()
        )

    def set_component(self, component):
        id_mapping = {
            "img2img_batch_input_dir": "img2img_batch_input_dir",
            "img2img_batch_output_dir": "img2img_batch_output_dir",
            "txt2img_generate": "txt2img_submit_button",
            "img2img_generate": "img2img_submit_button",
            "txt2img_width": "txt2img_w_slider",
            "txt2img_height": "txt2img_h_slider",
            "img2img_width": "img2img_w_slider",
            "img2img_height": "img2img_h_slider",
            "img2img_img2img_tab": "img2img_img2img_tab",
            "img2img_img2img_sketch_tab": "img2img_img2img_sketch_tab",
            "img2img_batch_tab": "img2img_batch_tab",
            "img2img_inpaint_tab": "img2img_inpaint_tab",
            "img2img_inpaint_sketch_tab": "img2img_inpaint_sketch_tab",
            "img2img_inpaint_upload_tab": "img2img_inpaint_upload_tab",
            "img2img_inpaint_full_res": "img2img_inpaint_area",
            "txt2img_hr-checkbox": "txt2img_enable_hr",
        }
        elem_id = getattr(component, "elem_id", None)
        # Do not set component if it has already been set.
        # https://github.com/Mikubill/sd-webui-controlnet/issues/2587
        if elem_id in id_mapping and getattr(self, id_mapping[elem_id]) is None:
            setattr(self, id_mapping[elem_id], component)
            logger.debug(f"Setting {elem_id}.")
            logger.debug(
                f"A1111 initialized {sum(c is not None for c in vars(self).values())}/{len(vars(self).keys())}."
            )


class ControlNetUiGroup(object):
    refresh_symbol = "\U0001f504"  # 🔄
    switch_values_symbol = "\U000021C5"  # ⇅
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝
    clear_symbol = "\U0001F5D1️"  # 🗑️
    load_symbol = "\U0001F4C2"  # 📂
    tossup_1m_symbol = "⤴1M"  # ⤴1M
    tossdown_input_symbol = "⤵I"  # insert img2img source image
    tossdown_output_symbol = "⤵O"  # insert current output image
    close_tab_symbol = "✕"  # ✕ close the open Input tab
    move_tab_left_symbol = "←"  # move the open Input tab one place left

    tooltips = {
        "🔄": "Refresh",
        "✕": "Close this Input tab and drop its image (the last Input tab cannot be closed)",
        "←": "Move this Input tab one place left - the input order matters, "
             "e.g. multi-phase profiles give the leftmost input the unshifted wave",
        "🗑️": "Clear image",
        "📂": "Load image",
        "⤵I": "Insert the img2img source image (img2img only, needs a loaded source image)",
        "⤵O": "Insert the current output image (needs a generation result in the gallery)",
        "⤴1M": "Send 1Mpx dimensions to stable diffusion: keeps the aspect ratio of the (cropped) image, "
                    "targets 1024x1024 total pixels, rounded to multiples of 16",
        "\u2934": "Send dimensions to stable diffusion",
        "💥": "Run preprocessor",
        "📝": "New blank canvas on the open tab, at the current Width x Height",
        "📷": "Enable webcam",
        "⇄": "Mirror webcam",
    }

    a1111_context = A1111Context()
    # All ControlNetUiGroup instances created.
    all_ui_groups: List["ControlNetUiGroup"] = []

    @property
    def width_slider(self):
        if self.is_img2img:
            return ControlNetUiGroup.a1111_context.img2img_w_slider
        else:
            return ControlNetUiGroup.a1111_context.txt2img_w_slider

    @property
    def height_slider(self):
        if self.is_img2img:
            return ControlNetUiGroup.a1111_context.img2img_h_slider
        else:
            return ControlNetUiGroup.a1111_context.txt2img_h_slider

    def __init__(
        self,
        is_img2img: bool,
        default_unit: external_code.ControlNetUnit,
        photopea: Optional[Photopea] = None,
    ):
        # Whether callbacks have been registered.
        self.callbacks_registered: bool = False
        # Whether the render method on this object has been called.
        self.ui_initialized: bool = False

        self.is_img2img = is_img2img
        self.default_unit = default_unit
        self.photopea = photopea
        self.webcam_enabled = False
        self.webcam_mirrored = False

        # Note: All gradio elements declared in `render` will be defined as member variable.
        self.enabled = None
        self.upload_tab = None
        self.image = None
        self.generated_image_group = None
        self.generated_image = None
        self.mask_image_group = None
        self.mask_image = None
        self.output_mask_tab = None
        self.output_mask_canvas = None
        self.prompt_tab_p = None
        self.prompt_tab_n = None
        self.active_canvas = None
        self.preview_slot = None
        self.image_tabs = None
        self.input_tabs = []
        self.image_canvases = []
        self.input_enabled_checks = []
        self.input_slots_open = None
        self.add_input_tab = None
        self.input_order = None
        self.close_tab_button = None
        self.move_tab_left_button = None
        self.clear_image_button = None
        self.insert_input_image_button = None
        self.insert_output_image_button = None
        self.open_new_canvas_button = None
        self.load_image_button = None
        self.send_dimen_button = None
        self.send_1m_dimen_button = None
        self.pixel_perfect = None
        self.preprocessor_preview = None
        self.mask_upload = None
        self.weight_mask = None
        self.output_mask = None
        self.type_filter = None
        self.module = None
        self.trigger_preprocessor = None
        self.model = None
        self.refresh_models = None
        self.weight = None
        self.weight_profile = None
        self.guidance_start = None
        self.guidance_end = None
        self.advanced = None
        self.processor_res = None
        self.threshold_a = None
        self.threshold_b = None
        self.control_mode = None
        self.balance_accordion = None
        self.resize_mode = None
        self.use_preview_as_input = None
        self.openpose_editor = None
        self.image_upload_panel = None
        self.save_detected_map = None
        self.hr_option = None
        self.unit_prompt = None
        self.unit_negative_prompt = None
        self.unit_prompt_emb_strength = None
        self.unit_prompt_delta_scale = None
        self.unit_negative_prompt_emb_strength = None
        self.unit_negative_prompt_delta_scale = None
        self.unit_prompt_retention = None
        self.unit_prompt_retention_n = None
        # hidden channel carrying classify_controlnet_type of the selected
        # model; javascript reads it to gate band buttons / balance editor
        self.model_type_state = None
        self.balance_support_note = None
        # the unit gr.State, set in render(); read by register_run_annotator
        self.unit = None

        # Internal states for UI state pasting.
        self.prevent_next_n_module_update = 0
        self.prevent_next_n_slider_value_update = 0

        ControlNetUiGroup.all_ui_groups.append(self)

    def render(self, tabname: str, elem_id_tabname: str) -> None:
        """The pure HTML structure of a single ControlNetUnit. Calling this
        function will populate `self` with all gradio element declared
        in local scope.

        Args:
            tabname:
            elem_id_tabname:

        Returns:
            None
        """
        self.openpose_editor = OpenposeEditor()

        preview_check_elem_id = f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox"

        # Control Type / Preprocessor / Model selectors sit in a column left
        # of the image canvas, spread evenly over its height. The main option
        # checkboxes live hidden below them: 'enabled' remains part of the
        # unit but is driven by the checkbox in the accordion title (two-way
        # sync in javascript/active_units.js); the others are unused, but stay
        # registered so all the existing event wiring keeps working.
        with gr.Row(elem_classes=["controlnet_image_and_options"]):
            # NOTE: min_width here is inert for an open unit - style.css gives
            # .controlnet_main_options_column an !important flex basis/floor
            # that outranks the inline min-width gradio writes from this.
            with gr.Column(scale=0, min_width=300, elem_classes=["controlnet_main_options_column"]):
                self.type_filter = gr.Dropdown(
                    global_state.get_all_preprocessor_tags(),
                    label=f"Control Type",
                    value="All",
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_type_filter_dropdown",
                    elem_classes="controlnet_control_type_filter_dropdown",
                )
                with gr.Row(elem_classes=["controlnet_preprocessor_model"]):
                    self.module = gr.Dropdown(
                        global_state.get_all_preprocessor_names(),
                        label="Preprocessor (invert for black lines on white)",
                        value=self.default_unit.module,
                        scale=1,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
                    )
                    self.trigger_preprocessor = ToolButton(
                        value=ControlNetUiGroup.trigger_symbol,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor",
                        elem_classes=["cnet-run-preprocessor", "cnet-toolbutton"],
                        tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.trigger_symbol],
                    )
                with gr.Row(elem_classes=["controlnet_preprocessor_model"]):
                    self.model = gr.Dropdown(
                        global_state.get_all_controlnet_names(),
                        label=f"Model",
                        value=self.default_unit.model,
                        scale=1,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
                    )
                    self.refresh_models = ToolButton(
                        value=ControlNetUiGroup.refresh_symbol,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_refresh_models",
                        elem_classes=["cnet-toolbutton"],
                        tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.refresh_symbol],
                    )
                # The legacy Control Mode chooser is replaced by the per-step
                # balance profile editor; the field stays in the unit (API
                # back-compat) as a hidden constant.
                self.control_mode = gr.State(self.default_unit.control_mode.value)
                self.resize_mode = gr.Dropdown(
                    choices=[e.value for e in external_code.ResizeMode],
                    value=self.default_unit.resize_mode.value,
                    label="Resize Mode",
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio",
                    elem_classes="controlnet_resize_mode_radio",
                )
                # classified kind of the selected model ('controlnet' /
                # 'lllite' / ...): a hidden VIEW channel (not a unit field)
                # that weight_profile.js reads to gate the band buttons the
                # same way the prompt textboxes are gated here
                self.model_type_state = gr.Textbox(
                    value=global_state.classify_controlnet_type(self.default_unit.model),
                    visible=False,
                    elem_classes=["cnet-model-type-state"],
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_type_state",
                )
                with gr.Column(visible=False, elem_classes=["controlnet_hidden_options"]):
                    self.enabled = gr.Checkbox(
                        label="Enable",
                        value=self.default_unit.enabled,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_enable_checkbox",
                        elem_classes=["cnet-unit-enabled"],
                    )
                    self.pixel_perfect = gr.Checkbox(
                        label="Pixel Perfect",
                        value=self.default_unit.pixel_perfect,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
                    )
                    self.preprocessor_preview = gr.Checkbox(
                        label="Allow Preview",
                        value=False,
                        elem_classes=["cnet-allow-preview"],
                        elem_id=preview_check_elem_id,
                    )
                    self.mask_upload = gr.Checkbox(
                        label="Use Mask",
                        value=False,
                        elem_classes=["cnet-mask-upload"],
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_mask_upload_checkbox",
                    )
                    self.use_preview_as_input = gr.Checkbox(
                        label="Preview as Input",
                        value=False,
                        elem_classes=["cnet-preview-as-input"],
                        visible=False,
                    )

            with gr.Column(scale=1):
                # Always shown, on img2img too: an empty canvas simply falls back
                # to the main img2img image (see get_input_data), and the copy
                # button covers the "same input" case explicitly.
                with gr.Group() as self.image_upload_panel:
                    # Which canvas the below-canvas button row acts on. Kept in
                    # sync by the tab select events below; the client-side
                    # buttons resolve the same thing from the visible tab panel.
                    self.active_canvas = gr.State("input_0")
                    # Which input slot the preprocessor preview was made from
                    # (run_annotator previews the OPEN tab). Only read when
                    # use_preview_as_input is on, to pick that slot's weight
                    # masks - a State, so it costs nothing to carry.
                    self.preview_slot = gr.State(0)
                    # One boolean per input slot: which Input tabs are open.
                    # Only the UI cares - the backend simply uses every slot
                    # that holds an image, and closing a tab clears its canvas.
                    self.input_slots_open = gr.State(
                        [slot == 0 for slot in range(external_code.MAX_INPUT_IMAGES)]
                    )
                    self.input_tabs = []
                    self.image_canvases = []
                    self.input_enabled_checks = []
                    with gr.Tabs(visible=True, elem_classes=["cnet-image-tabs"]) as self.image_tabs:
                        for slot in range(external_code.MAX_INPUT_IMAGES):
                            input_image_elem_id = (
                                f"{elem_id_tabname}_{tabname}_input_image" if slot == 0
                                else f"{elem_id_tabname}_{tabname}_input_image_{slot}"
                            )
                            with gr.Tab(
                                label="In",  # kept short: the injected mute checkbox already widens the tab
                                visible=(slot == 0),
                                id=f"cnet_input_{slot}",
                                elem_id=f"{elem_id_tabname}_{tabname}_input_tab_{slot}",
                            ) as input_tab:
                                with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                                    # Every input canvas hosts the full weight
                                    # mask toolset; the slot class tells
                                    # weight_mask.js which hidden channels
                                    # (cnet-wmask-<slot>-<key>) are its own, so
                                    # each input's masks stay independent.
                                    with gr.Group(elem_classes=[
                                            "cnet-input-image-group", f"cnet-input-slot-{slot}"]):
                                        canvas = ForgeCanvas(
                                            elem_id=input_image_elem_id,
                                            elem_classes=["cnet-image"],
                                            no_scribbles=True,  # mask scribbling superseded by the weight mask tool
                                            height=300,
                                            numpy=True
                                        )
                                        if slot == 0:
                                            self.openpose_editor.render_upload()
                                    if slot == 0:
                                        self.render_preview_groups(elem_id_tabname, tabname)
                            self.input_tabs.append(input_tab)
                            self.image_canvases.append(canvas)

                    # kept as the canonical names: everything that only ever
                    # dealt with "the" input image still points at slot 1
                    self.image = self.image_canvases[0]
                    self.upload_tab = self.input_tabs[0]

                    with self.image_tabs:
                        with gr.Tab(
                            label="+",
                            id="cnet_add_input",
                            elem_id=f"{elem_id_tabname}_{tabname}_add_input_tab",
                        ) as self.add_input_tab:
                            gr.HTML(
                                value='<p class="cnet-add-input-hint">Opens another Input tab for this unit. '
                                      'Every input that holds an image is preprocessed on its own, and the '
                                      'resulting controls are summed into this unit\'s output.</p>',
                            )

                        # Output-side weight mask. The image loaded here is a
                        # throwaway drawing backdrop (a rough layout, a previous
                        # result, ...) - only the painted mask is read, and it is
                        # registered with the GENERATED image, not with the
                        # control input. Nothing painted = control applies to the
                        # whole output, exactly as with no mask at all.
                        with gr.Tab(
                            label="Output mask",
                            id="cnet_output_mask",
                            # gradio names the strip button "<elem_id>-button";
                            # javascript/tab_marks.js needs it to mark this tab
                            elem_id=f"{elem_id_tabname}_{tabname}_output_mask_tab",
                        ) as self.output_mask_tab:
                            with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                                with gr.Group(elem_classes=["cnet-output-mask-group"]):
                                    self.output_mask_canvas = ForgeCanvas(
                                        elem_id=f"{elem_id_tabname}_{tabname}_output_mask_image",
                                        elem_classes=["cnet-image"],
                                        no_scribbles=True,
                                        height=300,
                                        numpy=True
                                    )

                        # Per-unit prompts for the control branch (true
                        # ControlNet models only; empty = the main prompts),
                        # each on its own tab right of Output mask so they
                        # cost NO permanent space. The labels are unique and
                        # hidden: an EMPTY label collides in ui-config.json
                        # ("txt2img//value" - one such stale entry injected
                        # "(presets)" into the box); enabled/disabled follows
                        # the selected MODEL's type (unit_prompt_state /
                        # register_unit_prompt_support).
                        prompt_state = ControlNetUiGroup.unit_prompt_state(self.default_unit.model)
                        negative_state = ControlNetUiGroup.unit_prompt_state(self.default_unit.model, "negative")
                        with gr.Tab(
                            label="P",
                            id="cnet_prompt_p",
                            elem_id=f"{elem_id_tabname}_{tabname}_prompt_tab_p",
                        ) as self.prompt_tab_p:
                            self.unit_prompt = gr.Textbox(
                                label="ControlNet positive prompt",
                                show_label=False,
                                value=self.default_unit.unit_prompt,
                                lines=5,
                                max_lines=5,
                                placeholder=prompt_state["placeholder"],
                                interactive=prompt_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_prompt_textbox",
                                elem_classes=["controlnet_unit_prompt"],
                            )
                            # strength pair of the positive unit prompt (see
                            # the dataclass comments): embedding strength
                            # moves the control context toward/past the text,
                            # effect scale multiplies the text's residual
                            # delta (0/1 stay single-pass, anything else
                            # costs a second control forward per step)
                            self.unit_prompt_emb_strength = gr.Slider(
                                label="P embedding strength",
                                value=self.default_unit.unit_prompt_emb_strength,
                                minimum=-1.0, maximum=3.0, step=0.05,
                                interactive=prompt_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_prompt_emb_strength",
                                elem_classes=["controlnet_unit_prompt_slider"],
                            )
                            self.unit_prompt_delta_scale = gr.Slider(
                                label="P effect scale",
                                value=self.default_unit.unit_prompt_delta_scale,
                                minimum=-1.0, maximum=3.0, step=0.05,
                                interactive=prompt_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_prompt_delta_scale",
                                elem_classes=["controlnet_unit_prompt_slider"],
                            )
                            # prompt retention: ONE global knob for the P and
                            # N sides alike. The unit prompt's influence
                            # naturally decays over the steps as the latent
                            # converges; retention r ramps the effect scale by
                            # 1 + r * progress so the text keeps steering in
                            # later steps. 0 = off (natural decay). This is
                            # the CANONICAL component (the unit field); the N
                            # tab shows a mirror kept in sync (identical
                            # label on purpose: they share one ui-config
                            # default, correct for a mirrored knob).
                            self.unit_prompt_retention = gr.Slider(
                                label="Prompt retention (P and N)",
                                value=self.default_unit.unit_prompt_retention,
                                minimum=0.0, maximum=3.0, step=0.05,
                                interactive=prompt_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_prompt_retention",
                                elem_classes=["controlnet_unit_prompt_slider",
                                              "controlnet_unit_prompt_slider_full"],
                            )
                        # negative counterpart: replaces the sampled negative
                        # context on the control branch's uncond rows ("push
                        # this control away from X"); same model gating
                        with gr.Tab(
                            label="N",
                            id="cnet_prompt_n",
                            elem_id=f"{elem_id_tabname}_{tabname}_prompt_tab_n",
                        ) as self.prompt_tab_n:
                            self.unit_negative_prompt = gr.Textbox(
                                label="ControlNet negative prompt",
                                show_label=False,
                                value=self.default_unit.unit_negative_prompt,
                                lines=5,
                                max_lines=5,
                                placeholder=negative_state["placeholder"],
                                interactive=negative_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_negative_prompt_textbox",
                                elem_classes=["controlnet_unit_prompt", "controlnet_unit_negative_prompt"],
                            )
                            # same strength pair for the negative side (the
                            # uncond rows of the control branch)
                            self.unit_negative_prompt_emb_strength = gr.Slider(
                                label="N embedding strength",
                                value=self.default_unit.unit_negative_prompt_emb_strength,
                                minimum=-1.0, maximum=3.0, step=0.05,
                                interactive=negative_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_negative_prompt_emb_strength",
                                elem_classes=["controlnet_unit_prompt_slider"],
                            )
                            self.unit_negative_prompt_delta_scale = gr.Slider(
                                label="N effect scale",
                                value=self.default_unit.unit_negative_prompt_delta_scale,
                                minimum=-1.0, maximum=3.0, step=0.05,
                                interactive=negative_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_negative_prompt_delta_scale",
                                elem_classes=["controlnet_unit_prompt_slider"],
                            )
                            # MIRROR of the P tab's retention slider (that one
                            # is the unit field; this is display/input only,
                            # synced both ways in register_unit_prompt_support)
                            self.unit_prompt_retention_n = gr.Slider(
                                label="Prompt retention (P and N)",
                                value=self.default_unit.unit_prompt_retention,
                                minimum=0.0, maximum=3.0, step=0.05,
                                interactive=prompt_state["interactive"],
                                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_unit_prompt_retention_n",
                                elem_classes=["controlnet_unit_prompt_slider",
                                              "controlnet_unit_prompt_slider_full"],
                            )

                    if self.photopea:
                        self.photopea.attach_photopea_output(self.generated_image.background)

                    # The below-canvas button row drives whichever canvas tab is
                    # open. The client-side buttons find it by walking every
                    # canvas and taking the one that is actually laid out
                    # (gradio hides the inactive tab panel, so offsetParent of
                    # everything inside it is null); the server-side ones read
                    # self.active_canvas, kept in sync by the tab select events.
                    canvas_root_ids = [
                        (f"{elem_id_tabname}_{tabname}_input_image" if slot == 0
                         else f"{elem_id_tabname}_{tabname}_input_image_{slot}")
                        for slot in range(external_code.MAX_INPUT_IMAGES)
                    ] + [f"{elem_id_tabname}_{tabname}_output_mask_image"]

                    def forward_to_active_canvas(button_prefix):
                        # The rule itself lives in javascript/active_canvas.js
                        # (window.cnetForwardToActiveCanvas), next to
                        # window.cnetVisible which it uses - this only hands it
                        # DATA. Logic written as a python f-string is invisible
                        # to every JS tool and needs a server restart to change.
                        roots = json.dumps(canvas_root_ids)
                        return (f'function(){{ window.cnetForwardToActiveCanvas('
                                f'{roots}, "{button_prefix}"); }}')

                    with gr.Row(elem_classes="controlnet_image_controls"):
                        # No static text anymore (the [invert] hint moved into
                        # the Preprocessor label), but the element stays: it is
                        # the host of the raster-info line (image_info.js
                        # writes into '.controlnet_invert_warning p') and it
                        # fills the row, pushing the tab actions right.
                        gr.HTML(
                            value="<p></p>",
                            elem_classes="controlnet_invert_warning",
                        )
                        # Per-slot mute: unchecking skips this input at
                        # generation time (image and masks are kept), so inputs
                        # can be A/B-ed without destructively clearing the
                        # canvas. These checkboxes are pure HIDDEN state
                        # channels: gradio cannot render components inside the
                        # tab strip, so the visible control - a native checkbox
                        # before each Input tab's title text - is injected and
                        # kept in sync by javascript/tab_marks.js (which also
                        # grays a muted tab's label).
                        for slot in range(external_code.MAX_INPUT_IMAGES):
                            self.input_enabled_checks.append(gr.Checkbox(
                                value=True,
                                label="use input",
                                visible=False,
                                elem_id=f"{elem_id_tabname}_{tabname}_input_enabled_{slot}",
                                elem_classes=["cnet-input-enabled"],
                            ))
                        # Visual order of the Input tabs (slot digits, "" =
                        # natural), a hidden state channel like the mute
                        # checkboxes: tab_marks.js mirrors it onto the strip
                        # buttons as CSS order, get_input_data sorts by it.
                        # Content stays in its slot; only this string moves.
                        self.input_order = gr.Textbox(
                            value="",
                            label="input order",
                            visible=False,
                            elem_id=f"{elem_id_tabname}_{tabname}_input_order",
                            elem_classes=["cnet-input-order-state"],
                        )
                        # closes the open Input tab (never the last one, and
                        # never the Output mask tab - see
                        # register_canvas_tab_events); starts inert because the
                        # unit opens with a single input
                        self.close_tab_button = ToolButton(
                            value=ControlNetUiGroup.close_tab_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_close_tab_button",
                            elem_classes=["cnet-toolbutton", "cnet-close-tab"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.close_tab_symbol],
                            interactive=False,
                        )
                        # moves the open Input tab one visual place left (the
                        # input order is generation-relevant, see the
                        # input_order channel above); inert until the open tab
                        # has an open left neighbor
                        self.move_tab_left_button = ToolButton(
                            value=ControlNetUiGroup.move_tab_left_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_move_tab_left_button",
                            elem_classes=["cnet-toolbutton", "cnet-move-tab-left"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.move_tab_left_symbol],
                            interactive=False,
                        )
                        # replaces the canvas toolbar Remove button (hidden on
                        # both canvases via style.css - toolbar diet); the click
                        # is forwarded to that still-bound core button, so
                        # clearing behaves exactly as before
                        self.clear_image_button = ToolButton(
                            value=ControlNetUiGroup.clear_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_clear_image_button",
                            elem_classes=["cnet-toolbutton"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.clear_symbol],
                        )
                        self.clear_image_button.click(
                            fn=None,
                            _js=forward_to_active_canvas("removeButton_"),
                            show_progress=False,
                            inputs=[],
                            outputs=[],
                        )
                        # insert-image buttons: fully client-side (see
                        # javascript/insert_image.js) - the click reads the
                        # source pixels in the browser and pushes them into
                        # this unit's input canvas; the same script keeps the
                        # disabled state in sync with source availability
                        # (rendered disabled here so they start inert)
                        self.insert_input_image_button = ToolButton(
                            value=ControlNetUiGroup.tossdown_input_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_insert_input_image_button",
                            elem_classes=["cnet-toolbutton", "cnet-insert-input-image"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossdown_input_symbol],
                            interactive=False,
                        )
                        self.insert_output_image_button = ToolButton(
                            value=ControlNetUiGroup.tossdown_output_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_insert_output_image_button",
                            elem_classes=["cnet-toolbutton", "cnet-insert-output-image"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossdown_output_symbol],
                            interactive=False,
                        )
                        self.open_new_canvas_button = ToolButton(
                            value=ControlNetUiGroup.open_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button",
                            elem_classes=["cnet-toolbutton"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.open_symbol],
                        )
                        # replaces the canvas toolbar Upload button (hidden on
                        # the input canvas via style.css), same click-forward
                        # pattern as Clear above
                        self.load_image_button = ToolButton(
                            value=ControlNetUiGroup.load_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_load_image_button",
                            elem_classes=["cnet-toolbutton"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.load_symbol],
                        )
                        self.load_image_button.click(
                            fn=None,
                            _js=forward_to_active_canvas("uploadButton_"),
                            show_progress=False,
                            inputs=[],
                            outputs=[],
                        )
                        self.send_dimen_button = ToolButton(
                            value=ControlNetUiGroup.tossup_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_dimen_button",
                            elem_classes=["cnet-toolbutton"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossup_symbol],
                        )
                        self.send_1m_dimen_button = ToolButton(
                            value=ControlNetUiGroup.tossup_1m_symbol,
                            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_1m_dimen_button",
                            elem_classes=["cnet-toolbutton"],
                            tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossup_1m_symbol],
                        )

                    # tab open/close/select wiring is registered at the end of
                    # render() (see there): closing a tab clears the slot's
                    # weight-mask channels too, and those are only built further
                    # down, next to the profile editors


        def profile_editor_html(label, hint, bands=False):
            # One shared editor markup for the weight and the balance profile;
            # javascript/weight_profile.js attaches the same editor class to
            # every '.cnet-weight-profile-editor' block.
            # bands: the weight editor additionally carries six thin selector
            # buttons at the bottom of the presets column, in TWO GROUPS split by
            # a separator. Exactly one is pressed, and that selection is the
            # MODE, not just an edit target: it decides what the unit runs on,
            # which is also the only thing drawn.
            #
            #   main | depth | drift   ASSEMBLE. None of them replaces another:
            #     the unit runs main(step) x depth(layer - drift(step)), so
            #     pressing any of the three runs the same field and only changes
            #     which factor is on screen. Depth multiplies the main profile;
            #     drift moves WHERE the depth curve is read as sampling proceeds,
            #     which is the only thing that couples the step and depth axes.
            #   coarse | mid | fine    REPLACE. Each is a whole per-step curve
            #     for its third of the same depth axis the depth curve covers
            #     continuously, so the two are alternatives - a per-bucket curve
            #     times a per-depth curve would count depth twice.
            #
            # The separator carries that distinction and nothing else does, so it
            # is a real element rather than a margin: reading order is the model,
            # and the flat row it replaced said all of them were siblings.
            #
            # The editor writes the pressed selector into the profile string as
            # '#B<band>' so the mode survives reloads (the three main-mode ones
            # write no marker - main is what runs for all of them); python reads
            # it with external_code.band_mode_active.
            #
            # Colors come from the shared CSS variables (style.css :root), the
            # same ones weight_profile.js resolves for the plot lines - one
            # source, no drift between button and line color.
            band_buttons = (
                '<div class="cnet-profile-bands">'
                '<button type="button" class="cnet-profile-band cnet-profile-band-active"'
                ' data-band="main" style="--band-color:#ffffff"'
                ' title="Main weight profile: per-step strength of the whole unit.'
                ' Selected = this is the profile the unit runs on; select a band'
                ' below the separator to run on the band profiles instead.'
                ' Its spatial half is the G weight mask on the canvas toolbar: while main'
                ' (or depth, or drift) is selected, G is the mask that applies and C/M/F'
                ' are dormant."></button>'
                # Depth profile: the same depth axis the three bands quantize,
                # un-quantized. It does NOT replace the main profile the way a
                # band does - it multiplies it, so the unit runs on
                # main(step) x depth(layer). Purple keeps it visually apart
                # from the three band colors.
                '<button type="button" class="cnet-profile-band" data-band="depth"'
                ' style="--band-color:var(--cnet-depth-line, #9c4dff)"'
                ' title="Depth profile (D): per-LAYER multiplier over UNet depth'
                ' (left = fine/texture, right = coarse/composition).'
                ' Unlike a band it does not replace the main profile - it multiplies it,'
                ' so the unit runs on main(step) x depth(layer). Flat 1 = off.'
                ' Its own plot, so the range selects switch to the depth multiplier range'
                ' (default 0..2, neutral 1 in the middle) while it is selected.'
                ' Bands and depth are alternatives: selecting a band runs the band profiles instead.'
                ' Because depth multiplies MAIN, the G weight mask is the one that applies while'
                ' it is selected, exactly as in main mode."></button>'
                # Depth-DRIFT profile: the third degree of freedom of the
                # main x depth pair. Without it that product is separable, so the
                # depth shape is frozen in time - the one thing the band profiles
                # could express and it could not. Green, and a plot of its own
                # again because its neutral is 0 rather than the multiplier 1.
                '<button type="button" class="cnet-profile-band" data-band="drift"'
                ' style="--band-color:var(--cnet-drift-line, #00c853)"'
                ' title="Depth-drift profile (S): per-STEP shift of the depth profile along'
                ' the depth axis, so the unit runs main(step) x depth(layer - drift(step)).'
                ' X is the sampling step like the main profile; Y is the shift - up moves the'
                ' depth curve toward coarse/composition, down toward fine/texture. A descending'
                ' curve therefore sweeps the control from composition to texture as sampling'
                ' proceeds. Flat 0 = off, and it does nothing at all unless a depth curve is'
                ' drawn for it to move.'
                ' Its own plot (default -1..1, neutral 0 in the middle).'
                ' This is the one thing main x depth cannot otherwise express: without a drift'
                ' the depth shape cannot change while sampling. The bands buy that same freedom'
                ' by quantizing depth into three buckets; this keeps depth continuous."></button>'
                # The group boundary. Above: three curves that multiply into one
                # field. Below: three that each replace it.
                '<div class="cnet-profile-band-sep" aria-hidden="true"></div>'
                '<button type="button" class="cnet-profile-band" data-band="coarse"'
                ' style="--band-color:var(--cnet-band-coarse, #e53935)"'
                ' title="Coarse band profile (C): per-step strength of the deepest injection'
                ' layers - composition. Its spatial half is the C weight mask on the canvas'
                ' toolbar, which multiplies it where painted.'
                ' While a band is selected the unit runs on the band profiles rather than the'
                ' main one, and on the C/M/F masks rather than G."></button>'
                '<button type="button" class="cnet-profile-band" data-band="mid"'
                ' style="--band-color:var(--cnet-band-mid, #fdd835)"'
                ' title="Mid band profile (M): per-step strength of the middle injection'
                ' layers - form. Its spatial half is the M weight mask on the canvas toolbar,'
                ' which multiplies it where painted.'
                ' While a band is selected the unit runs on the band profiles rather than the'
                ' main one, and on the C/M/F masks rather than G."></button>'
                '<button type="button" class="cnet-profile-band" data-band="fine"'
                ' style="--band-color:var(--cnet-band-fine, #1e88e5)"'
                ' title="Fine band profile (F): per-step strength of the shallowest injection'
                ' layers - texture. Its spatial half is the F weight mask on the canvas'
                ' toolbar, which multiplies it where painted.'
                ' While a band is selected the unit runs on the band profiles rather than the'
                ' main one, and on the C/M/F masks rather than G."></button>'
                '</div>'
            ) if bands else ''
            return (
                '<div class="cnet-weight-profile">'
                f'<span class="cnet-weight-profile-label">{label}'
                f' <span class="cnet-weight-profile-hint">{hint}</span></span>'
                '<div class="cnet-weight-profile-body">'
                '<div class="cnet-weight-profile-presets">'
                '<button type="button" class="cnet-profile-preset" data-preset="step"'
                ' title="Step preset. Click once to aim the parameter pad at it (the profile is kept),'
                ' click again to rebuild the profile from the pad. Pad x = jump position;'
                ' pad y = height AND direction - above the pad&#39;s middle the raised part is on the'
                ' right, below it on the left (the mirrored step), and the distance from the middle'
                ' is the height (middle row = flat 0)">'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"'
                ' stroke-linecap="round" stroke-linejoin="round"><path d="M3 19 H12 V5 H21"/></svg></button>'
                # ONE oscillatory button, cycled rather than two toggles that
                # forced each other on: multi-phase always needed the wave, so
                # "wave off + multi on" was never a reachable state and the
                # pair only ever expressed one ladder. It now IS one ladder,
                # and the icon is the whole of the state.
                #   off -> cosine -> multi-cosine -> multi-Fejer -> multi-von Mises -> off
                # The multi rungs are weight-editor only (they distribute the
                # wave over the unit's Inputs; the balance profile has none to
                # distribute over) - same gate as the band selectors, and
                # without it the cycle is just off -> cosine -> off.
                + ('<button type="button" class="cnet-profile-preset cnet-profile-osc"'
                   ' data-preset="osc" data-osc-state="off"'
                   + (' data-osc-multi="1"' if bands else '')
                   + ' title="Oscillatory mode, cycled by clicking: off → cosine →'
                   + (' multi-phase cosine → multi-phase Fejér → multi-phase von Mises →' if bands else '')
                   + ' off. The drawn profile becomes the ENVELOPE of the wave (still editable'
                   ' point by point). Pad x = phase 0..2π (von Mises: sharpness κ 0..10),'
                   ' pad y = 0..4 oscillations.'
                   + (' Multi-phase splits that one wave between the Inputs - input 1 as drawn,'
                      ' each next shifted by 2π/n - so they take turns steering across the steps.'
                      ' Fejér and von Mises share the envelope exactly (their weights sum to 1 at'
                      ' every step); von Mises adds κ, from 0 = every input equally on to 10 ='
                      ' near-hard switching. Thin lines preview the sibling waves. With a single'
                      ' input the multi-phase rungs change nothing.' if bands else '')
                   + '">'
                   # one icon per rung, shown by data-osc-state (see style.css).
                   # 'off' shows the plain wave, unlit - the same reading the
                   # cosine toggle had when it was its own button.
                   '<span data-osc="cos">'
                   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"'
                   ' stroke-linecap="round" stroke-linejoin="round">'
                   '<path d="M3 12 Q6 4 9 12 T15 12 T21 12"/></svg></span>'
                   + ('<span data-osc="multi">'
                      # three stacked waves: reads as "several lines" at button
                      # size (the old two overlapping half-opacity waves read as
                      # one smudged wave)
                      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"'
                      ' stroke-linecap="round" stroke-linejoin="round">'
                      '<path d="M3 6 Q7.5 2 12 6 T21 6"/>'
                      '<path d="M3 12 Q7.5 8 12 12 T21 12"/>'
                      '<path d="M3 18 Q7.5 14 12 18 T21 18"/></svg></span>'
                      # the two named families are lettered rather than drawn:
                      # their lobes differ from the cosine's by shape alone,
                      # which no 18px glyph can carry, and F/M are exactly how
                      # the tooltip and the docs name them
                      '<span data-osc="fejer" class="cnet-profile-osc-glyph">F</span>'
                      '<span data-osc="mises" class="cnet-profile-osc-glyph">M</span>'
                      if bands else '')
                   + '</button>')
                # (no invert toggle: the step preset spans both directions
                # through the pad's vertical halves, which is what invert was
                # used for - see applyPadPreset in weight_profile.js)
                + f'{band_buttons}'
                '</div>'
                '<div class="cnet-profile-preset-pad" title="Preset parameters (click a preset first)">'
                '<span class="cnet-profile-pad-dot"></span></div>'
                '<canvas class="cnet-weight-profile-canvas"></canvas>'
                # scale range as two selects instead of the old two-handle
                # gutter slider: top one is the range maximum, bottom one the
                # minimum. Options are filled in by weight_profile.js so the
                # -1..2 / 0.25 grid lives in exactly one place. The pair is the
                # PLOT's Y axis, not the selected curve's - and the editor has
                # two plots: the step axis (main + the three bands, drawn
                # together) and the depth axis (the depth curve alone, a
                # per-layer multiplier). The selects show whichever is on
                # screen; they never reach across.
                '<div class="cnet-profile-scale">'
                '<select class="cnet-profile-scale-hi"'
                ' title="Range top: the plot value 1 maps to this weight.'
                ' The range belongs to the plot on screen - in main/band mode it applies'
                ' to the main profile and all three band profiles at once, in depth mode'
                ' to the depth multiplier alone."></select>'
                # response exponent: vertical slider filling the free middle
                # of the range column. Bends the NORMALIZED profile (y -> y^e)
                # BEFORE the range mapping, so it is independent of the two
                # selects around it; the wrapper's center tick marks e = 1.
                '<div class="cnet-profile-gamma-wrap"'
                ' title="Response exponent: bends the normalized profile y -> y^e before the'
                ' range mapping. Middle = linear (e = 1, the tick). Up: e -> 1/10, values pushed'
                ' toward the top (10th root). Down: e -> 10, values pushed toward the bottom'
                ' (10th power). Applies to the selected profile line, waves included.'
                ' Double-click resets to linear.">'
                '<input type="range" class="cnet-profile-gamma" min="-100" max="100"'
                ' step="1" value="0">'
                '</div>'
                '<select class="cnet-profile-scale-lo"'
                ' title="Range bottom: the plot value 0 maps to this weight'
                ' (below 0 = repulsive control).'
                ' The range belongs to the plot on screen - in main/band mode it applies'
                ' to the main profile and all three band profiles at once, in depth mode'
                ' to the depth multiplier alone."></select>'
                '</div>'
                '</div>'
                '</div>'
            )

        default_profile = self.default_unit.weight_profile or external_code.weight_profile_from_scalars(
            self.default_unit.weight,
            self.default_unit.guidance_start,
            self.default_unit.guidance_end,
        )
        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            gr.HTML(
                value=profile_editor_html(
                    'Control Weight Profile',
                    '(X: relative step, Y: strength;'
                    ' click: add/move point, double-click: delete point;'
                    ' drag green midpoint: bend segment, double-click it: straighten)',
                    bands=True,
                ),
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_weight_profile_editor",
                elem_classes=["cnet-weight-profile-editor"],
            )
            # Hidden channel between the canvas editor (javascript/weight_profile.js)
            # and the backend. Holds the profile serialized as 'x@y;x@y;...'.
            self.weight_profile = gr.Textbox(
                value=default_profile,
                visible=False,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_weight_profile",
                elem_classes=["cnet-weight-profile-state"],
            )
            self.weight = gr.State(self.default_unit.weight)
            self.guidance_start = gr.State(self.default_unit.guidance_start)
            self.guidance_end = gr.State(self.default_unit.guidance_end)
            # Hidden channels for the rainbow-hue weight masks painted directly
            # on an input image canvas via its toolbar tools (weight_mask.js);
            # each holds the full-alpha mask PNG at that input's resolution. Per
            # input slot: a global mask (priority over the bands) plus the three
            # per-band layer masks. The painter finds them by the class, which
            # carries the slot it belongs to.
            self.weight_masks = []
            for slot in range(external_code.MAX_INPUT_IMAGES):
                channels = {}
                for band in (None, "coarse", "mid", "fine"):
                    field = external_code.ControlNetUnit.weight_mask_field(slot, band)
                    channels[band or "global"] = LogicalImage(
                        visible=False,
                        label=field,
                        numpy=True,
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_{field}",
                        elem_classes=[f"cnet-wmask-{slot}-{band or 'global'}-state"],
                    )
                self.weight_masks.append(channels)
            # canonical names for the first input, still referenced by name
            self.weight_mask = self.weight_masks[0]["global"]
            self.weight_mask_coarse = self.weight_masks[0]["coarse"]
            self.weight_mask_mid = self.weight_masks[0]["mid"]
            self.weight_mask_fine = self.weight_masks[0]["fine"]
            # Hidden channel for the mask painted on the "Output mask" tab
            # (same painter, same hue encoding); read as an output-side
            # injection mask only - it never gates the control input.
            self.output_mask = LogicalImage(
                visible=False,
                label='output_mask',
                numpy=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_output_mask",
                elem_classes=["cnet-output-mask-state"],
            )

        # Per-step cond/uncond balance, replacing the legacy Control Mode
        # chooser: y = 0.5 balanced, 1 = control matters most, 0 = prompt
        # matters most. Hidden for preprocessors whose patcher does not
        # implement balance (show_control_mode flag, e.g. reference).
        with gr.Accordion(label="Control Balance Profile", open=False,
                          elem_classes=["controlnet_balance_accordion"]) as self.balance_accordion:
            # model-type gate: shown when the selected MODEL's patcher ignores
            # the balance profile (ControlLLLite), so the user learns it here
            # instead of from a runtime log line after drawing a dead curve
            self.balance_support_note = gr.HTML(
                value="",
                visible=False,
                elem_classes=["cnet-balance-support-note"],
            )
            with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
                gr.HTML(
                    value=profile_editor_html(
                        'Control Balance Profile',
                        '(X: relative step, Y: balance; 0.5 = balanced,'
                        ' 1 = control matters most, 0 = prompt matters most;'
                        ' drag green midpoint: bend segment, double-click it: straighten)',
                    ),
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_balance_profile_editor",
                    elem_classes=["cnet-weight-profile-editor", "cnet-balance-profile-editor"],
                )
                self.balance_profile = gr.Textbox(
                    value=self.default_unit.balance_profile or "0@0.5;1@0.5",
                    visible=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_balance_profile",
                    elem_classes=["cnet-weight-profile-state"],
                )

        # advanced options (sliders side by side to save vertical space)
        with gr.Row(visible=False) as self.advanced:
            self.processor_res = gr.Slider(
                label="Preprocessor resolution",
                # -1 is the unit's "resolve at generation time" sentinel, and it
                # is not a legal slider value (minimum is 64). Show CNPro's
                # default instead; an explicitly-set value still wins, so a unit
                # restored from infotext keeps its own resolution.
                value=(self.default_unit.processor_res
                       if self.default_unit.processor_res > 0
                       else global_state.DEFAULT_PROCESSOR_RES),
                minimum=64,
                maximum=2048,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider",
            )
            self.threshold_a = gr.Slider(
                label="Threshold A",
                value=self.default_unit.threshold_a,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider",
            )
            self.threshold_b = gr.Slider(
                label="Threshold B",
                value=self.default_unit.threshold_b,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider",
            )

        with gr.Row(elem_classes=["controlnet_modes", "controlnet_row"]):
            self.hr_option = gr.Radio(
                choices=[e.value for e in HiResFixOption],
                value=self.default_unit.hr_option.value,
                label="Hires-Fix Option",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_hr_option_radio",
                elem_classes="controlnet_hr_option_radio",
                visible=False,
            )

        # Gradio calls UiControlNetUnit POSITIONALLY with these components, so
        # their order must equal the ControlNetUnit dataclass field order. The
        # mapping is written as name -> component and checked against the
        # dataclass below, so a mismatch fails loudly at startup instead of
        # silently shifting values into wrong fields.
        unit_fields = {
            "use_preview_as_input": self.use_preview_as_input,
            "generated_image": self.generated_image.background,
            "mask_image": self.mask_image.background,
            "mask_image_fg": self.mask_image.foreground,
            "hr_option": self.hr_option,
            "enabled": self.enabled,
            "module": self.module,
            "model": self.model,
            "weight": self.weight,
            "image": self.image.background,
            "image_fg": self.image.foreground,
            **{
                key: component
                for slot, canvas in enumerate(self.image_canvases[1:], start=2)
                for key, component in ((f"image_{slot}", canvas.background),
                                       (f"image_{slot}_fg", canvas.foreground))
            },
            **{
                ("image_enabled" if slot == 0 else f"image_{slot + 1}_enabled"): check
                for slot, check in enumerate(self.input_enabled_checks)
            },
            "resize_mode": self.resize_mode,
            "processor_res": self.processor_res,
            "threshold_a": self.threshold_a,
            "threshold_b": self.threshold_b,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
            "pixel_perfect": self.pixel_perfect,
            "control_mode": self.control_mode,
            "weight_profile": self.weight_profile,
            **{
                external_code.ControlNetUnit.weight_mask_field(slot, band): channels[band or "global"]
                for slot, channels in enumerate(self.weight_masks)
                for band in (None, "coarse", "mid", "fine")
            },
            "output_mask": self.output_mask,
            "balance_profile": self.balance_profile,
            "unit_prompt": self.unit_prompt,
            "unit_negative_prompt": self.unit_negative_prompt,
            "unit_prompt_emb_strength": self.unit_prompt_emb_strength,
            "unit_prompt_delta_scale": self.unit_prompt_delta_scale,
            "unit_negative_prompt_emb_strength": self.unit_negative_prompt_emb_strength,
            "unit_negative_prompt_delta_scale": self.unit_negative_prompt_delta_scale,
            "unit_prompt_retention": self.unit_prompt_retention,
            "input_order": self.input_order,
            "preview_slot": self.preview_slot,
        }
        all_fields = [f.name for f in dataclasses.fields(UiControlNetUnit)]
        expected_fields = all_fields[:len(unit_fields)]
        trailing_fields = all_fields[len(unit_fields):]
        # trailing fields silently take their defaults on the positional
        # rebuild, so they are pinned explicitly: a new dataclass field
        # appended without a component fails here instead of defaulting.
        # A hard raise, not assert: python -O strips asserts and this check is
        # the only thing standing between a field mismatch and silently
        # shifted unit values.
        if list(unit_fields) != expected_fields or trailing_fields != ["save_detected_map"]:
            raise RuntimeError(
                f"unit_fields is out of sync with the ControlNetUnit dataclass:\n"
                f"  components: {list(unit_fields)}\n"
                f"  dataclass:  {expected_fields}\n"
                f"  trailing (defaults-only, must be exactly ['save_detected_map']): {trailing_fields}"
            )

        unit = gr.State(self.default_unit)
        # the tab strip wiring needs the weight-mask channels built above, and
        # register_run_annotator reads the unit State - keep both reachable
        self.unit = unit
        self.register_canvas_tab_events()

        # Editing ONE field must not re-upload the whole unit. The obvious
        # wiring (fn=UiControlNetUnit, inputs=list(unit_args)) makes gradio
        # serialize every input on every event, so a single brush stroke on a
        # weight mask also re-sent the input images, the other masks and the
        # preview - measured at 3.8x the mask's own size with one input loaded,
        # and it grew with every further input. A gr.State input is kept
        # server-side and costs nothing to pass, so each component instead
        # patches its own field into the existing unit and uploads only the
        # value that actually changed.
        # WRITE ONE FIELD, NOT A WHOLE UNIT. `dataclasses.replace` builds a NEW
        # unit out of the snapshot it was handed, so it does not write one field
        # - it rewrites all ~60 from values read when THAT request started. Two
        # of these overlapping is a lost update by construction: the later
        # writer's snapshot predates the earlier writer's field, and reverts it.
        #
        # That is not hypothetical, and it is not rare. ONE raster landing on a
        # canvas moves two channels in the same tick (measured: background and
        # foreground write within the same 5 ms) - so a drop fires an `image`
        # change carrying megabytes of base64 alongside an `image_fg` change
        # carrying a blank canvas. The small one finishes first, the big one
        # lands second... or the small one lands second on a busy server and
        # puts the OLD image back. The channel and the canvas are then correct
        # and the generation still uses the previous picture, permanently -
        # until something rewrites the field, which is why closing and
        # reopening the tab "fixed" it (close_active_slot writes the unit
        # atomically; nothing else did).
        #
        # `setattr` on the live State object touches exactly the one attribute
        # it owns. Concurrent handlers write disjoint attributes and cannot
        # revert each other, and attribute assignment needs no lock. gr.State
        # passes its value through by reference (State.preprocess/postprocess
        # are identity), so the object mutated here IS the stored one.
        def field_updater(field_name):
            def update(value, current):
                if not dataclasses.is_dataclass(current):
                    return current
                setattr(current, field_name, value)
                return current
            return update

        for field_name, comp in unit_fields.items():
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)

            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=field_updater(field_name), inputs=[comp, unit], outputs=unit,
                    show_progress=False,
                )

        # The Generate click re-reads only the LIGHT fields (scalars, strings,
        # dropdowns) from their components and takes every image and mask from
        # the State. The split is deliberate:
        # - light fields come from the components because paths that bypass
        #   events could exist for them (ui-config.json restore writes initial
        #   values at build time, for one) and they cost nothing to upload;
        # - heavy fields (LogicalImage channels: input images, weight masks,
        #   preview, Use-Mask, output mask) are ONLY ever written through
        #   event-firing paths - user uploads, painter strokes, server pushes
        #   (gradio 4 fires .change for those too) - so the State already holds
        #   them and re-uploading potentially tens of MB of base64 on every
        #   Generate bought nothing but latency.
        light_fields = {
            name: comp for name, comp in unit_fields.items()
            if not isinstance(comp, LogicalImage)
        }

        # In place, for the same reason field_updater is: this fires on the
        # Generate click, which is exactly when a just-dropped image's `image`
        # change is still in flight. Rebuilding the unit from a snapshot here
        # would revert it, and the generation would run on the old picture.
        def merge_light_fields(*args):
            *values, current = args
            base = current if dataclasses.is_dataclass(current) else copy.copy(self.default_unit)
            for name, value in zip(light_fields, values):
                setattr(base, name, value)
            return base

        (
            ControlNetUiGroup.a1111_context.img2img_submit_button
            if self.is_img2img
            else ControlNetUiGroup.a1111_context.txt2img_submit_button
        ).click(
            fn=merge_light_fields,
            inputs=list(light_fields.values()) + [unit],
            outputs=unit,
            queue=False,
        )
        self.register_core_callbacks()
        self.ui_initialized = True
        return unit

    def render_preview_groups(self, elem_id_tabname: str, tabname: str) -> None:
        """Preprocessor preview + Use-Mask canvases, rendered next to the first
        input canvas only.

        Both are unit-level: there is one preview and one Use-Mask per unit, so
        they live in the first Input tab rather than being duplicated into every
        slot. Consequence worth knowing: the preview always RENDERS into the
        first tab, although run_annotator previews the image of whichever tab
        is open.
        """
        with gr.Group(
                visible=False, elem_classes=["cnet-generated-image-group"]
        ) as self.generated_image_group:
            self.generated_image = ForgeCanvas(
                elem_id=f"{elem_id_tabname}_{tabname}_generated_image",
                elem_classes=["cnet-image"],
                height=300,
                no_scribbles=True,
                no_upload=True,
                numpy=True
            )

            with gr.Group(elem_classes=["cnet-generated-image-control-group"]):
                if self.photopea:
                    self.photopea.render_child_trigger()
                self.openpose_editor.render_edit()
                preview_check_elem_id = f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox"
                preview_close_button_js = f"document.querySelector('#{preview_check_elem_id} input[type=\\'checkbox\\']').click();"
                gr.HTML(
                    value=f"""<a title="Close Preview" onclick="{preview_close_button_js}">Close</a>""",
                    visible=True,
                    elem_classes=["cnet-close-preview"],
                )

        with gr.Group(
                visible=False, elem_classes=["cnet-mask-image-group"]
        ) as self.mask_image_group:
            self.mask_image = ForgeCanvas(
                elem_id=f"{elem_id_tabname}_{tabname}_mask_image",
                elem_classes=["cnet-mask-image"],
                height=300,
                scribble_color='#FFFFFF',
                scribble_width=1,
                scribble_alpha_fixed=True,
                scribble_color_fixed=True,
                scribble_softness_fixed=True,
                numpy=True
            )

    def register_canvas_tab_events(self) -> None:
        """Open / close / select logic of the Input tab strip.

        The open set is a list of booleans rather than a count, so closing a tab
        in the middle never has to shuffle images between canvases: the closed
        slot is simply hidden and cleared, and the backend - which keys off "has
        an image", not off tab state - stops seeing it.
        """
        max_slots = external_code.MAX_INPUT_IMAGES

        def close_button_update(slots_open, active):
            # closing needs an input tab open and at least one other input tab
            # left to fall back to
            return gr.update(interactive=active.startswith("input_") and sum(slots_open) > 1)

        def open_slots_in_order(slots_open, order_value):
            """Open input slots, in the strip's visual order."""
            perm = external_code.ControlNetUnit.input_order_permutation(order_value)
            return [s for s in perm if slots_open[s]]

        def move_button_update(slots_open, active, order_value):
            # moving left needs an input tab open with an open input tab
            # visually before it
            if not active.startswith("input_"):
                return gr.update(interactive=False)
            slot = int(active[len("input_"):])
            visible = open_slots_in_order(slots_open, order_value)
            return gr.update(interactive=slot in visible and visible.index(slot) > 0)

        for slot, input_tab in enumerate(self.input_tabs):
            input_tab.select(
                fn=lambda slots_open, order_value, slot=slot: (
                    f"input_{slot}",
                    close_button_update(slots_open, f"input_{slot}"),
                    move_button_update(slots_open, f"input_{slot}", order_value),
                ),
                inputs=[self.input_slots_open, self.input_order],
                outputs=[self.active_canvas, self.close_tab_button, self.move_tab_left_button],
                show_progress=False,
            )

        self.output_mask_tab.select(
            fn=lambda: ("output_mask", gr.update(interactive=False), gr.update(interactive=False)),
            inputs=[],
            outputs=[self.active_canvas, self.close_tab_button, self.move_tab_left_button],
            show_progress=False,
        )

        # prompt tabs hold no canvas: the button row's server-side actions
        # (dims) see no image ("prompt" matches neither input_<n> nor
        # output_mask) and the close button stays inert
        for prompt_tab in (self.prompt_tab_p, self.prompt_tab_n):
            prompt_tab.select(
                fn=lambda: ("prompt", gr.update(interactive=False), gr.update(interactive=False)),
                inputs=[],
                outputs=[self.active_canvas, self.close_tab_button, self.move_tab_left_button],
                show_progress=False,
            )

        def open_next_slot(slots_open, order_value):
            slots_open = list(slots_open)
            target = next((i for i, open_ in enumerate(slots_open) if not open_), None)
            if target is None:
                # nothing left to open: bounce back to the last open tab
                target = max(i for i, open_ in enumerate(slots_open) if open_)
            slots_open[target] = True
            # the (re)opened tab appears at the visual END of the strip (next
            # to the "+" that opened it), wherever its slot's digit was left
            perm = external_code.ControlNetUnit.input_order_permutation(order_value)
            perm.remove(target)
            perm.append(target)
            order_value = "".join(str(s) for s in perm)
            active = f"input_{target}"
            return [
                slots_open,
                active,
                gr.update(selected=f"cnet_input_{target}"),
                close_button_update(slots_open, active),
                move_button_update(slots_open, active, order_value),
                order_value,
                gr.update(visible=not all(slots_open)),
            ] + [gr.update(visible=open_) for open_ in slots_open]

        self.add_input_tab.select(
            fn=open_next_slot,
            inputs=[self.input_slots_open, self.input_order],
            outputs=[
                self.input_slots_open,
                self.active_canvas,
                self.image_tabs,
                self.close_tab_button,
                self.move_tab_left_button,
                self.input_order,
                self.add_input_tab,
            ] + self.input_tabs,
            show_progress=False,
        )

        # a closed slot must not keep feeding the backend: its canvas
        # (background AND foreground - a stale scribble would resurrect as the
        # input the next time the slot gets a dark image), its four weight-mask
        # channels and its mute checkbox are all reset - a stale
        # mask surviving a close would silently gate whatever image the slot is
        # reopened with (the painter only auto-clears on a dimension change).
        # The unit State is ALSO patched here directly: the component clears
        # above echo back as 7 separate .change events, and waiting for all
        # seven to land before the slot is really empty makes the echoes
        # load-bearing. Writing the fields here makes them merely redundant.
        # (This used to be the ONLY atomic write into the unit, because
        # field_updater rebuilt the whole dataclass from a snapshot and could
        # revert a neighbour; it no longer can - see field_updater. That is
        # also why closing and reopening a tab was the only way to un-stick an
        # input image the racing writers had reverted.)
        mask_channels = [self.weight_masks[i][key]
                         for i in range(max_slots)
                         for key in ("global", "coarse", "mid", "fine")]
        close_outputs = [
            self.input_slots_open,
            self.active_canvas,
            self.image_tabs,
            self.close_tab_button,
            self.move_tab_left_button,
            self.add_input_tab,
            self.unit,
        ] + self.input_tabs \
          + [c.background for c in self.image_canvases] \
          + [c.foreground for c in self.image_canvases] \
          + mask_channels + self.input_enabled_checks

        def closed_slot_fields(slot):
            """unit field name -> cleared value for one closed slot."""
            image_field = "image" if slot == 0 else f"image_{slot + 1}"
            fields = {
                image_field: None,
                f"{image_field}_fg": None,
                f"{image_field}_enabled": True,
            }
            for band in (None, "coarse", "mid", "fine"):
                fields[external_code.ControlNetUnit.weight_mask_field(slot, band)] = None
            return fields

        def close_active_slot(slots_open, active, order_value, current_unit):
            slots_open = list(slots_open)
            slot = int(active[len("input_"):]) if active.startswith("input_") else -1
            if slot < 0 or not slots_open[slot] or sum(slots_open) <= 1:
                # guard the no-op cases: the button is meant to be disabled here
                return [gr.update()] * len(close_outputs)
            slots_open[slot] = False
            # fall back to the visual-left neighbor (the strip is ordered by
            # input_order, not by slot index), else the visual-first open tab
            perm = external_code.ControlNetUnit.input_order_permutation(order_value)
            fallback = next(
                (s for s in reversed(perm[:perm.index(slot)]) if slots_open[s]),
                next(s for s in perm if slots_open[s]))
            active = f"input_{fallback}"
            new_unit = current_unit
            if dataclasses.is_dataclass(new_unit):
                for name, value in closed_slot_fields(slot).items():
                    setattr(new_unit, name, value)
            return [
                slots_open,
                active,
                gr.update(selected=f"cnet_input_{fallback}"),
                close_button_update(slots_open, active),
                move_button_update(slots_open, active, order_value),
                gr.update(visible=not all(slots_open)),
                new_unit,
            ] + [gr.update(visible=open_) for open_ in slots_open] + [
                # canvas background of the closed slot
                gr.update(value=None) if i == slot else gr.update()
                for i in range(max_slots)
            ] + [
                # its foreground scribble layer
                gr.update(value=None) if i == slot else gr.update()
                for i in range(max_slots)
            ] + [
                # its four weight-mask channels
                gr.update(value=None) if i == slot else gr.update()
                for i in range(max_slots)
                for _ in range(4)
            ] + [
                # mute state channels (the visible checkbox lives in the tab
                # title, injected by tab_marks.js): reset the closed slot's,
                # so a reopened slot starts active
                gr.update(value=True) if i == slot else gr.update()
                for i in range(max_slots)
            ]

        self.close_tab_button.click(
            fn=close_active_slot,
            inputs=[self.input_slots_open, self.active_canvas, self.input_order, self.unit],
            outputs=close_outputs,
            show_progress=False,
        )

        # Move the open Input tab one visual place left: only the order
        # string changes hands - the tab keeps its slot (canvas, masks, mute
        # state stay put and stay selected), the strip re-sorts client-side
        # (tab_marks.js) and get_input_data feeds the backend in the new
        # order. That is what makes reordering non-destructive: no image or
        # mask ever travels between slots.
        def move_active_slot_left(slots_open, active, order_value):
            slot = int(active[len("input_"):]) if active.startswith("input_") else -1
            perm = external_code.ControlNetUnit.input_order_permutation(order_value)
            visible = [s for s in perm if slots_open[s]] if slot >= 0 else []
            if slot not in visible or visible.index(slot) == 0:
                # guard the no-op cases: the button is meant to be disabled here
                return [gr.update(), gr.update()]
            left = visible[visible.index(slot) - 1]
            a, b = perm.index(slot), perm.index(left)
            perm[a], perm[b] = perm[b], perm[a]
            order_value = "".join(str(s) for s in perm)
            return [order_value, move_button_update(slots_open, active, order_value)]

        self.move_tab_left_button.click(
            fn=move_active_slot_left,
            inputs=[self.input_slots_open, self.active_canvas, self.input_order],
            outputs=[self.input_order, self.move_tab_left_button],
            show_progress=False,
        )

    def canvas_backgrounds(self):
        """Background channels of every canvas the button row can address, in
        the order active_canvas_index expects."""
        return [canvas.background for canvas in self.image_canvases] + [
            self.output_mask_canvas.background
        ]

    def active_canvas_index(self, active):
        """Index into canvas_backgrounds() for an active_canvas key."""
        if active == "output_mask":
            return len(self.image_canvases)
        try:
            return int(str(active)[len("input_"):])
        except ValueError:
            return 0

    def register_send_dimensions(self):
        """Register event handler for send dimension button.

        Input-slot images come from the unit State: passing the LogicalImage
        components as inputs re-uploaded EVERY canvas image client->server on
        each click (tens of MB with several inputs loaded) just to read one
        shape. Heavy channels are only ever written through event-firing
        paths, so the State is authoritative for them (same rule as the
        Generate merge). The output-mask BACKDROP is not a unit field (only
        the painted mask is), so that one canvas stays a component input.
        """

        def active_image(active, unit, output_backdrop):
            """The image on the canvas tab the button row is pointing at."""
            if active == "output_mask":
                return output_backdrop
            if not active.startswith("input_") or not dataclasses.is_dataclass(unit):
                return None
            try:
                slot = int(active[len("input_"):])
            except ValueError:
                return None
            field = "image" if slot == 0 else f"image_{slot + 1}"
            return getattr(unit, field, None)

        def send_dimensions(active, unit, output_backdrop):
            def closesteight(num):
                rem = num % 8
                if rem <= 4:
                    return round(num - rem)
                else:
                    return round(num + (8 - rem))

            image = active_image(active, unit, output_backdrop)
            if image is not None:
                return closesteight(image.shape[1]), closesteight(image.shape[0])
            else:
                return gr.Slider.update(), gr.Slider.update()

        dimension_inputs = [self.active_canvas, self.unit, self.output_mask_canvas.background]

        self.send_dimen_button.click(
            fn=send_dimensions,
            inputs=dimension_inputs,
            outputs=[self.width_slider, self.height_slider],
            show_progress=False,
        )

        def send_dimensions_1m(active, unit, output_backdrop):
            """Same aspect ratio as the (cropped) input image, rescaled to a
            1Mpx (1024x1024) total, both sides rounded to multiples of 16."""
            def closest16(num):
                return max(16, int(round(num / 16)) * 16)

            image = active_image(active, unit, output_backdrop)
            if image is not None:
                h, w = image.shape[0], image.shape[1]
                scale = (1024 * 1024 / (w * h)) ** 0.5
                return closest16(w * scale), closest16(h * scale)
            else:
                return gr.Slider.update(), gr.Slider.update()

        self.send_1m_dimen_button.click(
            fn=send_dimensions_1m,
            inputs=dimension_inputs,
            outputs=[self.width_slider, self.height_slider],
            show_progress=False,
        )

    def register_refresh_all_models(self):
        def refresh_all_models():
            global_state.update_controlnet_filenames()
            return gr.Dropdown.update(
                choices=global_state.get_all_controlnet_names(),
            )

        self.refresh_models.click(
            refresh_all_models,
            outputs=[self.model],
            show_progress=False,
        )

    def register_build_sliders(self):
        def build_sliders(module: str, pp: bool):

            logger.debug(
                f"Prevent update slider value: {self.prevent_next_n_slider_value_update}"
            )
            logger.debug(f"Build slider for module: {module} - {pp}")

            preprocessor = global_state.get_preprocessor(module)

            slider_resolution_kwargs = preprocessor.slider_resolution.gradio_update_kwargs.copy()
            # CNPro default: 1024 for every preprocessor, replacing the host's
            # per-preprocessor value (512 from the Preprocessor base). Overridden
            # HERE as well as in the dataclass because this handler fires on every
            # preprocessor change and pushes the preprocessor's own value into the
            # slider - so setting only the dataclass default would give 1024 on
            # first render and silently drop back to 512 the moment the user
            # picked a module. Range and step still come from the preprocessor.
            slider_resolution_kwargs['value'] = global_state.default_processor_res(preprocessor)

            if pp:
                slider_resolution_kwargs['visible'] = False

            grs = [
                gr.update(**slider_resolution_kwargs),
                gr.update(**preprocessor.slider_1.gradio_update_kwargs.copy()),
                gr.update(**preprocessor.slider_2.gradio_update_kwargs.copy()),
                gr.update(visible=True),
                gr.update(visible=not preprocessor.do_not_need_model),
                gr.update(visible=not preprocessor.do_not_need_model),
                gr.update(visible=preprocessor.show_control_mode),
            ]

            return grs

        inputs = [
            self.module,
            self.pixel_perfect,
        ]
        outputs = [
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.advanced,
            self.model,
            self.refresh_models,
            self.balance_accordion,
        ]
        self.module.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )
        self.pixel_perfect.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )

        def filter_selected(k: str):
            logger.debug(f"Prevent update {self.prevent_next_n_module_update}")
            logger.debug(f"Switch to control type {k}")

            filtered_preprocessor_list = global_state.get_filtered_preprocessor_names(k)
            filtered_controlnet_names = global_state.get_filtered_controlnet_names(k)
            default_preprocessor = filtered_preprocessor_list[0]
            default_controlnet_name = filtered_controlnet_names[0]

            if k != 'All':
                if len(filtered_preprocessor_list) > 1:
                    default_preprocessor = filtered_preprocessor_list[1]
                if len(filtered_controlnet_names) > 1:
                    default_controlnet_name = filtered_controlnet_names[1]

            if self.prevent_next_n_module_update > 0:
                self.prevent_next_n_module_update -= 1
                return [
                    gr.Dropdown.update(choices=filtered_preprocessor_list),
                    gr.Dropdown.update(choices=filtered_controlnet_names),
                ]
            else:
                return [
                    gr.Dropdown.update(
                        value=default_preprocessor, choices=filtered_preprocessor_list
                    ),
                    gr.Dropdown.update(
                        value=default_controlnet_name, choices=filtered_controlnet_names
                    ),
                ]

        self.type_filter.change(
            fn=filter_selected,
            inputs=[self.type_filter],
            outputs=[self.module, self.model],
            show_progress=False,
        )

    def register_run_annotator(self):
        def run_annotator(active, unit, module, pres, pthr_a, pthr_b, t2i_w, t2i_h, pp, rm):
            # Preview the input of the OPEN tab, not always slot 1: the image
            # comes from the unit State (kept current by the per-field events),
            # which costs nothing to pass - re-uploading every canvas on each
            # preview click would not. On the Output mask tab the index runs
            # past the input slots and previewing falls through to "no image".
            image = None
            mask = None
            index = self.active_canvas_index(active)
            if dataclasses.is_dataclass(unit):
                slots = unit.input_images()
                if 0 <= index < len(slots):
                    image, mask, _enabled = slots[index]
            if image is None:
                return (
                    gr.update(visible=True),
                    None,
                    gr.update(),
                    index if 0 <= index < external_code.MAX_INPUT_IMAGES else 0,
                    *self.openpose_editor.update(""),
                )

            img = HWC3(image)
            # foreground can legitimately be absent (no_scribbles canvases):
            # HWC3 asserts on None, so guard before, not after
            mask = HWC3(mask) if mask is not None else None

            if mask is None or not (mask > 5).any():
                mask = None

            preprocessor = global_state.get_preprocessor(module)

            if pp:
                pres = external_code.pixel_perfect_resolution(
                    img,
                    target_H=t2i_h,
                    target_W=t2i_w,
                    resize_mode=external_code.resize_mode_from_value(rm),
                )

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = ""

                def accept(self, json_dict: dict) -> None:
                    self.value = json.dumps(json_dict)

            json_acceptor = JsonAcceptor()

            logger.info(f"Preview Resolution = {pres}")

            def is_openpose(module: str):
                return "openpose" in module

            # Only openpose preprocessor returns a JSON output, pass json_acceptor
            # only when a JSON output is expected. This will make preprocessor cache
            # work for all other preprocessors other than openpose ones. JSON acceptor
            # instance are different every call, which means cache will never take
            # effect.
            # TODO: Maybe we should let `preprocessor` return a Dict to alleviate this issue?
            # This requires changing all callsites though.
            result = preprocessor(
                input_image=img,
                resolution=pres,
                slider_1=pthr_a,
                slider_2=pthr_b,
                input_mask=mask,
                json_pose_callback=json_acceptor.accept
                if is_openpose(module)
                else None,
            )

            is_image = judge_image_type(result)

            if not is_image:
                result = img

            result = external_code.visualize_inpaint_mask(result)
            return (
                gr.update(visible=True),
                result,
                # preprocessor_preview
                gr.update(value=True),
                # which slot this preview came from: use-preview-as-input
                # replaces THAT slot's image, so it has to be gated by that
                # slot's weight masks rather than always by the first one's
                index if 0 <= index < external_code.MAX_INPUT_IMAGES else 0,
                # openpose editor
                *self.openpose_editor.update(json_acceptor.value),
            )

        self.trigger_preprocessor.click(
            fn=run_annotator,
            inputs=[
                self.active_canvas,
                self.unit,
                self.module,
                self.processor_res,
                self.threshold_a,
                self.threshold_b,
                self.width_slider,
                self.height_slider,
                self.pixel_perfect,
                self.resize_mode,
            ],
            outputs=[
                self.generated_image.block,
                self.generated_image.background,
                self.preprocessor_preview,
                self.preview_slot,
                *self.openpose_editor.outputs(),
            ],
        )

    def register_shift_preview(self):
        def shift_preview(is_on):
            return (
                # generated_image
                gr.update() if is_on else gr.update(value=None),
                # generated_image_group
                gr.update(visible=is_on),
                # use_preview_as_input,
                gr.update(visible=False),  # Now this is automatically managed
                # download_pose_link
                gr.update() if is_on else gr.update(value=None),
                # modal edit button
                gr.update() if is_on else gr.update(visible=False),
            )

        self.preprocessor_preview.change(
            fn=shift_preview,
            inputs=[self.preprocessor_preview],
            outputs=[
                self.generated_image.background,
                self.generated_image_group,
                self.use_preview_as_input,
                self.openpose_editor.download_link,
                self.openpose_editor.modal,
            ],
            show_progress=False,
        )

    def register_create_canvas(self):
        canvas_backgrounds = self.canvas_backgrounds()

        def fn_canvas(active, h, w):
            # Size comes from the main Width/Height sliders, so a new canvas is
            # always born at the generation resolution - which is what the
            # output mask wants (it is registered with the generated image) and
            # what a from-scratch control drawing wants too.
            blank = np.zeros(shape=(h, w, 3), dtype=np.uint8)
            target = self.active_canvas_index(active)
            return [blank if i == target else gr.update()
                    for i in range(len(canvas_backgrounds))]

        self.open_new_canvas_button.click(
            fn=fn_canvas,
            inputs=[self.active_canvas, self.height_slider, self.width_slider],
            outputs=canvas_backgrounds,
            show_progress=False,
        )

    def register_shift_hr_options(self):
        ControlNetUiGroup.a1111_context.txt2img_enable_hr.change(
            fn=lambda checked: gr.update(visible=checked),
            inputs=[ControlNetUiGroup.a1111_context.txt2img_enable_hr],
            outputs=[self.hr_option],
            show_progress=False,
        )

    def register_shift_upload_mask(self):
        """Controls whether the upload mask input should be visible."""
        def on_checkbox_click(checked: bool, canvas_height: int, canvas_width: int):
            if not checked:
                # Clear mask_image if unchecked.
                return gr.update(visible=False), gr.update(value=None)
            else:
                # Init an empty canvas the same size as the generation target.
                empty_canvas = np.zeros(shape=(canvas_height, canvas_width, 3), dtype=np.uint8)
                return gr.update(visible=True), gr.update(value=empty_canvas)

        self.mask_upload.change(
            fn=on_checkbox_click,
            inputs=[self.mask_upload, self.height_slider, self.width_slider],
            outputs=[self.mask_image_group, self.mask_image.background],
            show_progress=False,
        )

    def register_clear_preview(self):
        def clear_preview(x):
            if x:
                logger.info("Preview as input is cancelled.")
            return gr.update(value=False), gr.update(value=None)

        # ForgeCanvas is a WRAPPER, not a gradio component: it has no .edit,
        # .click, .change or .clear, so listing the canvases themselves here
        # registered NOTHING and every input image change left the preview
        # standing. Upstream this loop got gr.Image objects, which carried
        # .edit/.clear; the migration to ForgeCanvas broke it silently, and the
        # hasattr cascade below is exactly the shape that cannot report it.
        # Measured against a live instance before the fix: run the preprocessor
        # on a 200x120 input, drop a 320x240 raster - the input channel updates,
        # the preview panel keeps showing the 200x120 result.
        # Subscribe to the canvases' real channels instead; the scribble layer
        # counts too, since it feeds the preprocessor as well.
        canvas_channels = [
            channel
            for canvas in self.image_canvases
            for channel in (canvas.background, canvas.foreground)
        ]
        for comp in (
            self.pixel_perfect,
            self.module,
            *canvas_channels,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
        ):
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)
            if not event_subscribers:
                # the silent-no-op mode this whole block just came back from:
                # a component with no matching event subscribes to nothing and
                # says nothing. Fail at startup instead.
                raise RuntimeError(
                    "register_clear_preview: %r exposes none of edit/click/"
                    "release/change, so the preprocessor preview would never be "
                    "invalidated when it changes." % (type(comp).__name__,)
                )
            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=clear_preview,
                    inputs=self.use_preview_as_input,
                    outputs=[self.use_preview_as_input, self.generated_image.background],
                    show_progress=False
                )

    # display names for the unit-prompt hint; keep in sync with
    # global_state.classify_controlnet_type
    UNIT_PROMPT_TYPE_LABELS = {
        "lllite": "ControlLLLite",
        "ipadapter": "IP-Adapter",
        "t2i": "T2I-Adapter",
    }

    @staticmethod
    def unit_prompt_state(model_name, role="positive"):
        """interactive + placeholder of a unit-prompt textbox for a model.

        Only true ControlNets (ControlNet / ControlLora - encoder copies with
        their own text cross-attention) consume a unit prompt. Other types
        keep the textbox VISIBLE but disabled, with a placeholder saying which
        types support it - the hint is the education/discoverability, hiding
        would erase both. 'unknown' fails open (classic .pth ControlNets).
        The two boxes share one label, so when ENABLED the placeholder is what
        names the role (positive / negative).
        """
        kind = global_state.classify_controlnet_type(model_name)
        if kind in ("controlnet", "controllora", "unknown"):
            return {
                "interactive": True,
                "placeholder": ("positive - seen only by the control model"
                                if role == "positive"
                                else "negative - pushes this control away"),
            }
        if kind == "none":
            return {
                "interactive": False,
                "placeholder": "No model selected - ControlNet / ControlLora models can read"
                               " a per-unit prompt with their own text attention",
            }
        label = ControlNetUiGroup.UNIT_PROMPT_TYPE_LABELS.get(kind, kind)
        return {
            "interactive": False,
            "placeholder": f"{label} models have no text input - the unit prompt is read"
                           f" by ControlNet / ControlLora models",
        }

    # model kinds whose patcher honors the balance profile: ControlNetPatcher
    # (controlnet / controllora / t2i all load through it) and IPAdapterPatcher.
    # 'unknown' fails open, 'none' = nothing selected yet (no nagging).
    BALANCE_SUPPORT_KINDS = ("controlnet", "controllora", "t2i", "ipadapter", "unknown", "none")

    def register_unit_prompt_support(self):
        def update_model_type(model_name):
            kind = global_state.classify_controlnet_type(model_name)
            balance_ok = kind in ControlNetUiGroup.BALANCE_SUPPORT_KINDS
            label = ControlNetUiGroup.UNIT_PROMPT_TYPE_LABELS.get(kind, kind)
            note = ("" if balance_ok else
                    f"<p>{label} models ignore the balance profile - the curve below"
                    f" will have no effect with this model.</p>")
            positive_state = ControlNetUiGroup.unit_prompt_state(model_name, "positive")
            negative_state = ControlNetUiGroup.unit_prompt_state(model_name, "negative")
            # the strength sliders share the textboxes' gate (they modify what
            # the prompt does, so they are equally meaningless without it)
            slider_gate = gr.update(interactive=positive_state["interactive"])
            return (
                gr.update(**positive_state),
                gr.update(**negative_state),
                slider_gate, slider_gate, slider_gate, slider_gate, slider_gate, slider_gate,
                kind,
                gr.update(value=note, visible=not balance_ok),
            )

        self.model.change(
            fn=update_model_type,
            inputs=[self.model],
            outputs=[self.unit_prompt, self.unit_negative_prompt,
                     self.unit_prompt_emb_strength, self.unit_prompt_delta_scale,
                     self.unit_negative_prompt_emb_strength, self.unit_negative_prompt_delta_scale,
                     self.unit_prompt_retention, self.unit_prompt_retention_n,
                     self.model_type_state, self.balance_support_note],
            show_progress=False,
        )

        # Retention mirror sync is CLIENT-SIDE (javascript/prompt_retention.js):
        # a server round-trip proved unreliable during slider drags (the N
        # mirror felt passively coupled and would not move). The JS couples the
        # two sliders' frontend values directly, so the P component - the unit
        # field - always carries whichever tab the user dragged.

    def register_core_callbacks(self):
        """Register core callbacks that only involves gradio components defined
        within this ui group."""
        self.register_refresh_all_models()
        self.register_build_sliders()
        self.register_unit_prompt_support()
        self.register_shift_preview()
        self.register_clear_preview()
        self.openpose_editor.register_callbacks(
            self.generated_image,
            self.use_preview_as_input,
            self.model,
        )
        assert self.type_filter is not None

    def register_callbacks(self):
        """Register callbacks that involves A1111 context gradio components."""
        # Prevent infinite recursion.
        if self.callbacks_registered:
            return

        self.callbacks_registered = True
        self.register_send_dimensions()
        self.register_run_annotator()
        # needs the main Width/Height sliders, so it belongs to the
        # A1111-context group rather than register_core_callbacks
        self.register_create_canvas()
        self.register_shift_upload_mask()
        if not self.is_img2img:
            self.register_shift_hr_options()

    @staticmethod
    def reset():
        ControlNetUiGroup.a1111_context = A1111Context()
        ControlNetUiGroup.all_ui_groups = []

    @staticmethod
    def try_register_all_callbacks():
        unit_count = shared.opts.data.get("control_net_unit_count", 3)
        all_unit_count = unit_count * 2  # txt2img + img2img.
        if (
            # All A1111 components ControlNet units care about are all registered.
            ControlNetUiGroup.a1111_context.ui_initialized
            and all_unit_count == len(ControlNetUiGroup.all_ui_groups)
            and all(
                g.ui_initialized and (not g.callbacks_registered)
                for g in ControlNetUiGroup.all_ui_groups
            )
        ):
            for ui_group in ControlNetUiGroup.all_ui_groups:
                ui_group.register_callbacks()

            logger.info("ControlNet UI callback registered.")

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        ControlNetUiGroup.a1111_context.set_component(component)
        ControlNetUiGroup.try_register_all_callbacks()
