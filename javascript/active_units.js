/**
 * Give a badge on ControlNet Accordion indicating total number of active
 * units.
 * Make active unit's tab name green.
 * Append control type to tab name.
 * Disable resize mode selection when A1111 img2img input is used.
 */
(function () {
    const cnetAllAccordions = new Set();
    onUiUpdate(() => {
        function childIndex(element) {
            // Get all child nodes of the parent
            let children = Array.from(element.parentNode.childNodes);

            // Filter out non-element nodes (like text nodes and comments)
            children = children.filter(child => child.nodeType === Node.ELEMENT_NODE);

            return children.indexOf(element);
        }

        function imageInputDisabledAlert() {
            alert('Inpaint control type must use a1111 input in img2img mode.');
        }

        class ControlNetUnitTab {
            constructor(tab, accordion) {
                this.tab = tab;
                this.tabOpen = false; // Whether the tab is open.
                this.accordion = accordion;
                this.isImg2Img = tab.querySelector('.cnet-mask-upload').id.includes('img2img');

                this.enabledAccordionCheckbox = tab.querySelector('.input-accordion-checkbox');
                this.enabledCheckbox = tab.querySelector('.cnet-unit-enabled input');
                // ALL input slots, not just the first: auto-enable, the run-
                // preprocessor button state and the collapsed thumbnail must
                // see every Input tab (a unit whose image lives in slot 3 is
                // just as loaded as one using slot 1)
                this.inputImageGroups = tab.querySelectorAll('.cnet-input-image-group');
                this.inputImageFiles = tab.querySelectorAll('.cnet-input-image-group .cnet-image input[type="file"]');
                this.generatedImageGroup = tab.querySelector('.cnet-generated-image-group');
                this.maskImageGroup = tab.querySelector('.cnet-mask-image-group');
                // first slot: the unit-level Use-Mask scribble lives there
                this.inputImageGroup = tab.querySelector('.cnet-input-image-group');
                this.controlTypeSelect = tab.querySelector('.controlnet_control_type_filter_dropdown input');
                this.runPreprocessorButton = tab.querySelector('.cnet-run-preprocessor');

                this.tabs = tab.parentNode;
                this.tabIndex = childIndex(tab);

                // By default the InputAccordion checkbox is linked with the state
                // of accordion's open/close state. To disable this link, we can
                // simulate click to check the checkbox and uncheck it.
                this.enabledAccordionCheckbox.click();
                this.enabledAccordionCheckbox.click();

                this.sync_enabled_checkbox();
                this.attachEnabledButtonListener();
                this.attachControlTypeRadioListener();
                this.attachImageUploadListener();
                this.attachImageStateChangeObserver();
                this.attachA1111SendInfoObserver();
                this.attachAccordionStateObserver();
            }

            /**
             * Sync the states of enabledCheckbox and enabledAccordionCheckbox.
             */
            sync_enabled_checkbox() {
                this.enabledCheckbox.addEventListener("change", () => {
                    if (this.enabledAccordionCheckbox.checked != this.enabledCheckbox.checked) {
                        this.enabledAccordionCheckbox.click();
                    }
                });
                this.enabledAccordionCheckbox.addEventListener("change", () => {
                    if (this.enabledCheckbox.checked != this.enabledAccordionCheckbox.checked) {
                        this.enabledCheckbox.click();
                    }
                });
            }
            /**
             * Get the span that has text "Unit {X}".
             */
            getUnitHeaderTextElement() {
                return this.tab.querySelector(
                    `button > span:nth-child(1)`
                );
            }

            getActiveControlType() {
                if (this.controlTypeSelect) {
                    return this.controlTypeSelect.value || null;
                }
                return null;
            }

            updateActiveState() {
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;

                if (this.enabledCheckbox.checked) {
                    unitHeader.classList.add('cnet-unit-active');
                } else {
                    unitHeader.classList.remove('cnet-unit-active');
                }
            }

            updateActiveUnitCount() {
                function getActiveUnitCount(checkboxes) {
                    let activeUnitCount = 0;
                    for (const checkbox of checkboxes) {
                        if (checkbox.checked)
                            activeUnitCount++;
                    }
                    return activeUnitCount;
                }

                const checkboxes = this.accordion.querySelectorAll('.cnet-unit-enabled input');
                const span = this.accordion.querySelector('.label-wrap span');

                // Remove existing badge.
                if (span.childNodes.length !== 1) {
                    span.removeChild(span.lastChild);
                }
                // Add new badge if necessary.
                const activeUnitCount = getActiveUnitCount(checkboxes);
                if (activeUnitCount > 0) {
                    const div = document.createElement('div');
                    div.classList.add('cnet-badge');
                    div.classList.add('primary');
                    div.innerHTML = `${activeUnitCount} unit${activeUnitCount > 1 ? 's' : ''}`;
                    span.appendChild(div);
                }
            }

            /**
             * Add the active control type to tab displayed text.
             */
            updateActiveControlType() {
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;

                // Remove the control if exists
                const controlTypeSuffix = unitHeader.querySelector('.control-type-suffix');
                if (controlTypeSuffix) controlTypeSuffix.remove();

                // Add new suffix. An emptied filter box (value "") must show
                // no suffix, not "[undefined]"; textContent because the
                // dropdown input carries free user text, never markup.
                const controlType = this.getActiveControlType();
                if (!controlType || controlType === 'All') return;

                const span = document.createElement('span');
                span.textContent = `[${controlType}]`;
                span.classList.add('control-type-suffix');
                unitHeader.appendChild(span);
            }
            getInputImageSrc() {
                // first slot that actually holds an image (the thumbnail used
                // to show only slot 1, leaving units driven by later slots
                // looking empty when collapsed)
                for (const group of this.inputImageGroups) {
                    const img = group.querySelector('.cnet-image .forge-image');
                    if (img && img.src.startsWith('data')) return img.src;
                }
                return null;
            }

            anyInputImageLoaded() {
                return this.getInputImageSrc() !== null;
            }

            updateRunButton() {
                const any = this.anyInputImageLoaded();
                if (this.runPreprocessorButton.hasAttribute('disabled') === !any) return;
                if (any) {
                    this.runPreprocessorButton.removeAttribute('disabled');
                    this.runPreprocessorButton.title = 'Run preprocessor';
                } else {
                    this.runPreprocessorButton.setAttribute('disabled', true);
                    this.runPreprocessorButton.title = 'No ControlNet input image available';
                }
            }
            getPreprocessorPreviewImageSrc() {
                const img = this.generatedImageGroup.querySelector('.cnet-image .forge-image');
                return (img && img.src.startsWith('data')) ? img.src : null;
            }
            getMaskImageSrc() {
                function isEmptyCanvas(canvas) {
                    if (!canvas) return true;
                    if (canvas.width == 0 || canvas.height ==0) return true;
                    const ctx = canvas.getContext('2d');
                    // Get the image data
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    const data = imageData.data; // This is a Uint8ClampedArray
                    // Check each pixel
                    let isPureBlack = true;
                    for (let i = 0; i < data.length; i += 4) {
                        if (data[i] !== 0 || data[i + 1] !== 0 || data[i + 2] !== 0) { // Check RGB values
                            isPureBlack = false;
                            break;
                        }
                    }
                    return isPureBlack;
                }
                const maskImg = this.maskImageGroup.querySelector('.cnet-mask-image .forge-image');
                // Hand-drawn mask on mask upload.
                const handDrawnMaskCanvas = this.maskImageGroup.querySelector('.cnet-mask-image .forge-drawing-canvas');
                // Hand-drawn mask on input image upload.
                const inputImageHandDrawnMaskCanvas = this.inputImageGroup.querySelector('.cnet-image .forge-drawing-canvas');
                if (!isEmptyCanvas(handDrawnMaskCanvas)) {
                    return handDrawnMaskCanvas.toDataURL();
                } else if (maskImg && maskImg.src.startsWith('data')) {
                    return maskImg.src;
                } else if (!isEmptyCanvas(inputImageHandDrawnMaskCanvas)) {
                    return inputImageHandDrawnMaskCanvas.toDataURL();
                } else {
                    return null;
                }
            }
            setThumbnail(imgSrc, maskSrc) {
                if (!imgSrc) return;
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;
                const img = document.createElement('img');
                img.src = imgSrc;
                img.classList.add('cnet-thumbnail');
                unitHeader.appendChild(img);

                if (maskSrc) {
                    const mask = document.createElement('img');
                    mask.src = maskSrc;
                    mask.classList.add('cnet-thumbnail');
                    unitHeader.appendChild(mask);
                }
            }
            removeThumbnail() {
                const unitHeader = this.getUnitHeaderTextElement();
                if (!unitHeader) return;
                const imgs = unitHeader.querySelectorAll('.cnet-thumbnail');
                for (const img of imgs) {
                    img.remove();
                }
            }
            /**
             * When the accordion is folded, display a thumbnail of input image
             * and mask on the accordion header.
             */
            updateInputImageThumbnail() {
                if (!opts.controlnet_input_thumbnail) return;
                if (this.tabOpen) {
                    this.removeThumbnail();
                } else {
                    this.setThumbnail(this.getInputImageSrc(), this.getMaskImageSrc());
                }
            }

            attachEnabledButtonListener() {
                this.enabledCheckbox.addEventListener('change', () => {
                    this.updateActiveState();
                    this.updateActiveUnitCount();
                });
            }

            attachControlTypeRadioListener() {
                if (this.controlTypeSelect) {
                    // Gradio dropdown sets the input value programmatically after
                    // these events fire (blur fires on mousedown, before an option
                    // click commits), so listen on the whole block in capture phase
                    // to catch option clicks and read the value one tick later.
                    const dropdownBlock = this.tab.querySelector('.controlnet_control_type_filter_dropdown');
                    for (const event of ['change', 'input', 'blur', 'click', 'keyup']) {
                        dropdownBlock.addEventListener(event, () => {
                            setTimeout(() => this.updateActiveControlType(), 0);
                        }, true);
                    }
                }
            }

            attachImageUploadListener() {
                // Automatically check `enable` checkbox when an image is
                // uploaded into ANY input slot.
                for (const fileInput of this.inputImageFiles) {
                    fileInput.addEventListener('change', (event) => {
                        if (!event.target.files) return;
                        if (!this.enabledCheckbox.checked)
                            this.enabledCheckbox.click();
                    });
                }

                // Automatically check `enable` checkbox when JSON pose file is uploaded.
                this.tab.querySelector('.cnet-upload-pose input').addEventListener('change', (event) => {
                    if (!event.target.files) return;
                    if (!this.enabledCheckbox.checked)
                        this.enabledCheckbox.click();
                });

                // Images can also arrive WITHOUT the file input: drag-drop,
                // paste, the insert buttons (forgeCanvasPush). The canvas
                // announcement covers them all; auto-enable only on the
                // empty -> filled TRANSITION, so re-announcements from
                // adjustments (crop, levels) can never re-enable a unit the
                // user deliberately disabled with an image loaded.
                this._hadAnyImage = this.anyInputImageLoaded();
                // document-level, one per tab INSTANCE - and a gradio
                // re-render creates a fresh instance, so the listener detaches
                // itself once its tab leaves the DOM instead of accumulating
                // (each held the whole detached subtree via `this`).
                const onImageInfo = (e) => {
                    if (!this.tab.isConnected) {
                        document.removeEventListener('forge-image-info', onImageInfo);
                        return;
                    }
                    if (!e.target || !this.tab.contains(e.target)) return;
                    this.updateRunButton();
                    const now = this.anyInputImageLoaded();
                    if (now && !this._hadAnyImage && !this.enabledCheckbox.checked) {
                        this.enabledCheckbox.click();
                    }
                    this._hadAnyImage = now;
                };
                document.addEventListener('forge-image-info', onImageInfo);
            }

            attachImageStateChangeObserver() {
                // one observer over every slot's canvas: the run-preprocessor
                // button reflects "any input holds an image", so clearing
                // slot 1 while slot 3 is loaded must keep it enabled
                const observer = new MutationObserver(() => this.updateRunButton());
                for (const group of this.inputImageGroups) {
                    const containerEl = group.querySelector('.cnet-image');
                    if (containerEl) {
                        observer.observe(containerEl, {
                            childList: true,
                            subtree: true,
                            attributes: true,
                            attributeFilter: ['src'],
                        });
                    }
                }
                this.updateRunButton();
            }

            /**
             * Observe send PNG info buttons in A1111, as they can also directly
             * set states of ControlNetUnit.
             */
            attachA1111SendInfoObserver() {
                const pasteButtons = gradioApp().querySelectorAll('#paste');
                const pngButtons = gradioApp().querySelectorAll(
                    this.isImg2Img ?
                        '#img2img_tab, #inpaint_tab' :
                        '#txt2img_tab'
                );

                for (const button of [...pasteButtons, ...pngButtons]) {
                    // self-removing for the same reason as onImageInfo above:
                    // the paste/send buttons are global and outlive every
                    // re-rendered tab instance that registered on them
                    const onClick = () => {
                        if (!this.tab.isConnected) {
                            button.removeEventListener('click', onClick);
                            return;
                        }
                        // The paste/send img generation info feature goes
                        // though gradio, which is pretty slow. Ideally we should
                        // observe the event when gradio has done the job, but
                        // that is not an easy task.
                        // Here we just do a 2 second delay until the refresh.
                        setTimeout(() => {
                            this.updateActiveState();
                            this.updateActiveUnitCount();
                        }, 2000);
                    };
                    button.addEventListener('click', onClick);
                }
            }

            /**
             * Observer that triggers when the ControlNetUnit's accordion(tab) closes.
             */
            attachAccordionStateObserver() {
                new MutationObserver((mutationsList) => {
                    for(const mutation of mutationsList) {
                        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                            const newState = mutation.target.classList.contains('open');
                            if (this.tabOpen != newState) {
                                this.tabOpen = newState;
                                if (newState) {
                                    this.onAccordionOpen();
                                } else {
                                    this.onAccordionClose();
                                }
                            }
                        }
                    }
                }).observe(this.tab.querySelector('.label-wrap'), { attributes: true, attributeFilter: ['class'] });
            }

            onAccordionOpen() {
                this.updateInputImageThumbnail();
            }

            onAccordionClose() {
                this.updateInputImageThumbnail();
                this.updateActiveControlType();
            }
        }

        gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
            if (cnetAllAccordions.has(accordion)) return;
            const tabs = [...accordion.querySelectorAll('.input-accordion')]
                .map(tab => new ControlNetUnitTab(tab, accordion));

            // On open of main extension accordion, if no unit is enabled,
            // open unit 0 for edit.
            const labelWrap = accordion.querySelector('.label-wrap');
            const observerAccordionOpen = new MutationObserver(function (mutations) {
                for (const mutation of mutations) {
                    if (mutation.target.classList.contains('open') &&
                        tabs.every(tab => !tab.enabledCheckbox.checked &&
                                          !tab.tab.querySelector('.label-wrap').classList.contains('open'))
                    ) {
                        tabs[0].tab.querySelector('.label-wrap').click();
                    }
                }
            });
            observerAccordionOpen.observe(labelWrap, { attributes: true, attributeFilter: ['class'] });

            cnetAllAccordions.add(accordion);
        });
    });
})();