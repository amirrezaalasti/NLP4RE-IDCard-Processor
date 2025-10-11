import re
import fitz
import json
import logging
from pathlib import Path

# Import mappings for better field label extraction
from .mappings import (
    resource_mappings,
    predicates_mapping,
    class_mappings,
    list_of_other_comments,
)


class PDFFormExtractor:
    """
    Extracts data from PDF forms, including finding text labels for interactive widgets.
    """

    def __init__(self, pdf_path: str, debug: bool = False):
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(f"No file found at {pdf_path}")
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(self.pdf_path)
        self.results = {}

        # Initialize mappings for better field extraction
        self.resource_mappings = resource_mappings
        self.predicates_mapping = predicates_mapping
        self.class_mappings = class_mappings
        self.list_of_other_comments = list_of_other_comments

        # Precompute an index from question_mapping tokens (e.g., 'II.1') to resource_mapping_key
        self._question_mapping_index = self._build_question_mapping_index()

        # logging setup
        self.debug = debug
        self.logger = logging.getLogger(__name__ + ".PDFFormExtractor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.info(
            f"Opened PDF '{self.pdf_path.name}' with {len(self.doc)} pages"
        )

    def extract_with_labels(self) -> dict:
        """
        Primary extraction method that prioritizes finding labels for interactive fields.
        """
        self.logger.info("Starting extraction of interactive form fields with labels")

        has_interactive_fields = any(page.widgets() for page in self.doc)

        if not has_interactive_fields:
            self.logger.warning("No interactive form fields found in this PDF")
            return {}

        # First, collect all raw field data
        raw_fields = self._collect_raw_field_data()

        # Then, structure it into questions with options and answers
        structured_data = self._structure_form_data(raw_fields)

        # Post-process to merge duplicate questions with same question_text
        structured_data = self._merge_duplicate_questions(structured_data)

        # Validate extracted data against mappings
        if self.resource_mappings:
            structured_data = self._validate_against_mappings(structured_data)

        self.results = structured_data
        self.logger.info(
            "Extraction complete: %d fields -> %d questions (%d with answers)",
            structured_data.get("extraction_summary", {}).get("total_fields_found", 0),
            structured_data.get("total_questions", 0),
            structured_data.get("extraction_summary", {}).get(
                "questions_with_selections", 0
            ),
        )
        return self.results

    def _collect_raw_field_data(self) -> list:
        """
        Collects all raw field data from the PDF.
        """
        all_fields = []

        for page in self.doc:
            words_on_page = page.get_text("words")

            for widget in page.widgets():
                widget_info = self._get_widget_info(widget, words_on_page)
                widget_info["page"] = page.number + 1
                all_fields.append(widget_info)
                if self.debug:
                    self.logger.debug(
                        "Collected widget | page=%s name=%s type=%s value=%s label=%s field_label=%s",
                        widget_info.get("page"),
                        widget_info.get("name"),
                        widget_info.get("type"),
                        widget_info.get("value"),
                        widget_info.get("label"),
                        widget_info.get("field_label"),
                    )

        return all_fields

    def _structure_form_data(self, raw_fields: list) -> dict:
        """
        Structures the raw field data into a more readable format with questions, options, and answers.
        """
        from collections import defaultdict
        import re

        # Group fields by their base question (removing the suffix parts like _0_, _1_, etc.)
        question_groups = defaultdict(list)

        for field in raw_fields:
            field_name = field["name"]
            if not field_name:
                continue

            # Extract the base question by removing suffixes like _0_, _1_, _edit;_, etc.
            base_question = re.sub(r"_\d+_[^_]*$|_edit;_[^_]*$", "", field_name)
            question_groups[base_question].append(field)

        # Structure the data
        structured_questions = []

        for base_question, fields in question_groups.items():
            if not fields:
                continue

            # Prefer field_label from any field; fallback to extracted text from name
            first_field_label = next(
                (f.get("field_label") for f in fields if f.get("field_label")), None
            )
            question_text = first_field_label or self._extract_question_text(
                base_question
            )

            # Resolve the resource mapping category for this question (context for label enhancement)
            resource_key = self._resolve_resource_key_for_question(question_text)

            # Branch schema by field type
            group_types = {f.get("type") for f in fields}
            if self.debug:
                self.logger.debug(
                    "Group base=%s types=%s derived_question_text='%s'",
                    base_question,
                    ",".join(sorted([t or "" for t in group_types])),
                    question_text,
                )
            # If it's a single Text field, treat as free-text answer
            if len(fields) == 1 and next(iter(group_types)) == "Text":
                text_field = fields[0]
                question_data = {
                    # "question_id": base_question,
                    "question_text": question_text,
                    "type": "Text",
                    "answer": text_field.get("value") or "",
                    "field_name": text_field.get("name"),
                }
                structured_questions.append(question_data)
                if self.debug:
                    self.logger.debug(
                        "Text question formed | base=%s field=%s answer='%s'",
                        base_question,
                        text_field.get("name"),
                        question_data.get("answer"),
                    )
                continue
            # Otherwise, assume choice-type (Radio / Checkbox) with options
            selected_options = []
            all_options = []

            # Get expected options from mappings to ensure completeness
            expected_options = self._get_expected_options_for_question(
                question_text, resource_key
            )
            found_option_labels = set()
            option_labels_to_info = (
                {}
            )  # key: normalized_label -> option_info (label kept as display)

            for field in fields:
                # Prefer the typed value for Text fields as the label when present
                text_value = None
                if field["type"] == "Text":
                    value = (field.get("value") or "").strip()
                    if not value:
                        # Skip empty text inputs entirely
                        continue
                    text_value = value

                # Enhance the field label using mappings with contextual resource key
                enhanced_label = self._enhance_label_with_mappings(
                    field["label"], resource_key
                )

                option_info = {
                    "label": text_value or enhanced_label,
                    "field_name": field["name"],
                    # Keep raw value for Text fields; clean others
                    "field_value": (
                        field.get("value")
                        if field.get("type") == "Text"
                        else self._clean_field_value(field.get("value"))
                    ),
                    "is_selected": self._is_field_selected(field),
                }
                # Preserve provenance when an option originates from a Text field
                if field.get("type") == "Text":
                    option_info["source_type"] = "Text"

                # For selected RadioButtons, prefer the widget's export value as label
                if (
                    field.get("type") == "RadioButton"
                    and option_info["is_selected"]
                    and field.get("value")
                ):
                    value_label = str(field.get("value") or "").strip()
                    # Only use export value if it looks like a human-readable label
                    if (
                        value_label
                        and value_label.lower() != "off"
                        and len(value_label) > 2
                        and re.search(r"[A-Za-z]", value_label)  # must contain a letter
                    ):
                        value_label = self._enhance_label_with_mappings(
                            value_label, resource_key
                        )
                        option_info["label"] = value_label

                option_label = option_info["label"]
                # Do NOT normalize Text-derived options to preserve user input uniqueness
                option_key = (
                    option_label
                    if field.get("type") == "Text"
                    else self._normalize_option_key(option_label)
                )

                # Handle duplicate labels by merging their information
                if option_key in option_labels_to_info:
                    # Merge with existing option - prefer selected state and combine field values
                    existing_info = option_labels_to_info[option_key]
                    if option_info["is_selected"]:
                        existing_info["is_selected"] = True
                    # If this field has a value and existing doesn't, or vice versa, combine them
                    if option_info["field_value"] and not existing_info.get(
                        "field_value"
                    ):
                        existing_info["field_value"] = option_info["field_value"]
                    # Do not concatenate multiple values; prefer the first meaningful one
                    # Prefer the longer, more informative display label
                    if len(option_label or "") > len(existing_info.get("label") or ""):
                        existing_info["label"] = option_label
                    if self.debug:
                        self.logger.debug(
                            "Merged duplicate option label | label='%s' existing_field=%s new_field=%s",
                            option_label,
                            existing_info["field_name"],
                            option_info["field_name"],
                        )
                else:
                    # New unique label
                    option_labels_to_info[option_key] = option_info
                    found_option_labels.add(option_label)

                if self.debug:
                    self.logger.debug(
                        "Option | base=%s name=%s type=%s value=%s label=%s enhanced=%s selected=%s",
                        base_question,
                        field.get("name"),
                        field.get("type"),
                        field.get("value"),
                        field.get("label"),
                        enhanced_label,
                        option_info.get("is_selected"),
                    )

            # Before finalizing, add any expected options missing from the PDF as synthetic options
            if expected_options:
                for expected_label in expected_options:
                    if expected_label in ["Not reported"]:
                        continue
                    expected_key = self._normalize_option_key(expected_label)
                    if expected_key not in option_labels_to_info:
                        option_labels_to_info[expected_key] = {
                            "label": expected_label,
                            "field_name": None,
                            "field_value": "",
                            "is_selected": False,
                            "source_type": "Mapping",
                        }

            # Convert the (possibly supplemented) options dictionary to list
            all_options = list(option_labels_to_info.values())

            # Rebuild selected_options from options after supplementation
            selected_options = []
            other_comments_text_found = None

            # First pass: collect all selected options and check for "Other/Comments" with text
            for _key, info in option_labels_to_info.items():
                if info.get("is_selected"):
                    field_value = info.get("field_value", "").strip()
                    label = info.get("label", "") or ""

                    # Check if this is an "Other/Comments" option with meaningful text content
                    is_other_comments = any(
                        other_comment.lower() in label.lower()
                        for other_comment in self.list_of_other_comments
                    )

                    if (
                        is_other_comments
                        and field_value
                        and len(field_value) > 3
                        and field_value.lower() not in ["off", "yes", "no"]
                    ):
                        # This is an "Other/Comments" option with actual comment text
                        other_comments_text_found = field_value
                    elif not is_other_comments:
                        # Only add non-"Other/Comments" options
                        selected_options.append(label)

            # If we found "Other/Comments" with text, use that instead of the generic label
            if other_comments_text_found:
                selected_options.append(other_comments_text_found)

            # Add missing expected options if mappings suggest they should be present
            if expected_options and self.debug:
                missing_options = set(expected_options) - found_option_labels
                if missing_options:
                    self.logger.debug(
                        "Question '%s' may be missing expected options: %s",
                        question_text,
                        list(missing_options)[:5],  # Show first 5
                    )

            # Create the structured question
            # Determine choice group type label
            group_type_label = (
                "RadioButton"
                if "RadioButton" in group_types
                else (
                    "CheckBox"
                    if "CheckBox" in group_types
                    else ",".join(sorted(group_types))
                )
            )
            question_data = {
                # "question_id": base_question,
                "question_text": question_text,
                "type": group_type_label,
                "selected_answers": selected_options if selected_options else ["None"],
                "all_options": [opt["label"] for opt in all_options],
                "options_details": all_options,
                "total_options": len(all_options),
            }

            structured_questions.append(question_data)
            if self.debug:
                self.logger.debug(
                    "Choice question formed | base=%s type=%s selected=%s total_options=%d",
                    base_question,
                    group_type_label,
                    ", ".join(selected_options) if selected_options else "None",
                    len(all_options),
                )

        # Derive summary counts with schema-aware logic
        def question_has_answer(q: dict) -> bool:
            qtype = q.get("type")
            if qtype == "Text":
                return bool(q.get("answer"))
            # Choice types
            selected = q.get("selected_answers")
            if selected is not None:
                return any(ans and ans != "None" for ans in selected)
            # Fallback using options_details
            for opt in q.get("options_details", []) or []:
                if opt.get("is_selected"):
                    return True
            return False

        questions_with_selections = sum(
            1 for q in structured_questions if question_has_answer(q)
        )

        result = {
            "pdf_name": self.pdf_path.name,
            "total_questions": len(structured_questions),
            "extraction_summary": {
                "total_fields_found": len(raw_fields),
                "questions_with_selections": questions_with_selections,
            },
            "questions": structured_questions,
        }
        if self.debug:
            self.logger.debug(
                "Structured %d questions (%d with answers) from %d fields",
                result["total_questions"],
                result["extraction_summary"]["questions_with_selections"],
                result["extraction_summary"]["total_fields_found"],
            )
        return result

    def _extract_question_text(self, base_question: str) -> str:
        """
        Extracts readable question text from the field name.
        """
        # Handle special cases first
        if base_question.startswith("_                             _"):
            return "Title and Authors"

        # Replace underscores with spaces and clean up
        question_text = base_question.replace("_", " ")

        # Remove hash-like suffixes (e.g., "3onV9GF51v2qn4B5z306pQ")
        question_text = re.sub(r"\s+[a-zA-Z0-9]{20,}\s*$", "", question_text)

        # Clean up Roman numeral patterns like "I 1 What RE Task..."
        question_text = re.sub(r"^(I+V*|V+I*)\s+(\d+)\s+", r"\1.\2. ", question_text)

        # Clean up multiple spaces
        question_text = re.sub(r"\s+", " ", question_text).strip()

        # Remove trailing incomplete words or artifacts
        question_text = re.sub(r"\s+[a-zA-Z]{1,3}$", "", question_text)

        # Capitalize first letter if it exists
        if question_text and len(question_text) > 1:
            question_text = question_text[0].upper() + question_text[1:]

        return question_text if question_text else "Question text not found"

    def _is_field_selected(self, field: dict) -> bool:
        """
        Determines if a field is selected based on its value and type.
        """
        field_value = field.get("value")
        field_type = field.get("type", "")

        if field_type == "RadioButton":
            # For radio buttons, check if value is not 'Off'
            selected = field_value not in ("Off", None, "")
            if self.debug:
                self.logger.debug(
                    "Selection check | type=RadioButton name=%s value=%s -> %s",
                    field.get("name"),
                    field_value,
                    selected,
                )
            return selected
        elif field_type == "CheckBox":
            # For checkboxes, check if value is not 'Off' or None
            selected = field_value not in ("Off", None, "")
            if self.debug:
                self.logger.debug(
                    "Selection check | type=CheckBox name=%s value=%s -> %s",
                    field.get("name"),
                    field_value,
                    selected,
                )
            return selected
        elif field_type == "Text":
            # For text fields, check if there's content
            selected = bool(field_value and field_value.strip())
            if self.debug:
                self.logger.debug(
                    "Selection check | type=Text name=%s value=%s -> %s",
                    field.get("name"),
                    field_value,
                    selected,
                )
            return selected

        return False

    def _get_widget_info(self, widget: fitz.Widget, words: list) -> dict:
        """
        Gets widget details and finds its associated text label.
        """
        widget_rect = widget.rect
        field_info = {
            "name": widget.field_name,
            "type": widget.field_type_string,
            "value": widget.field_value,
            # "rect": [round(c, 2) for c in widget_rect],
            "label": None,  # Default label
        }
        # Capture the form-defined field label if available (often holds the question text)
        try:
            field_info["field_label"] = widget.field_label
        except Exception:
            field_info["field_label"] = None

        # if widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
        #     field_info["is_checked"] = widget.is_checked
        # elif widget.field_type == fitz.PDF_WIDGET_TYPE_RADIOBUTTON:
        # # For radio buttons, is_checked tells if THIS specific button is selected
        # field_info["is_selected"] = widget.is_checked
        # # The group value shows which option was selected for the whole group
        # field_info["group_value"] = widget.field_value

        # Find the label for the widget using spatial analysis
        # Keep raw label here (no mappings applied yet) to avoid cross-category leakage
        field_info["label"] = self._find_label_for_widget(widget_rect, words)
        if self.debug:
            self.logger.debug(
                "Widget info | name=%s type=%s value=%s field_label=%s label=%s rect=(%.1f,%.1f,%.1f,%.1f)",
                field_info.get("name"),
                field_info.get("type"),
                field_info.get("value"),
                field_info.get("field_label"),
                field_info.get("label"),
                widget_rect.x0,
                widget_rect.y0,
                widget_rect.x1,
                widget_rect.y1,
            )

        return field_info

    def _find_label_for_widget(self, widget_rect: fitz.Rect, words: list) -> str:
        """
        Searches for text labels to the right of a given widget's rectangle.
        Uses both vertical alignment and horizontal proximity to avoid picking up distant text.

        Args:
            widget_rect: The fitz.Rect object for the form widget.
            words: A list of words on the page from page.get_text("words").

        Returns:
            The found text label as a string, or None if no label is found.
        """
        # Define tolerances for alignment and proximity
        VERTICAL_TOLERANCE = 3  # pixels for vertical alignment
        MAX_HORIZONTAL_DISTANCE = 160  # maximum pixels to look to the right (balanced to capture full options but avoid cross-column contamination)

        widget_mid_y = (widget_rect.y0 + widget_rect.y1) / 2

        # Find all words that are vertically aligned and close horizontally
        candidate_words = []
        for word_rect in words:
            x0, y0, x1, y1, word_text = word_rect[:5]
            word_mid_y = (y0 + y1) / 2

            # Check for vertical alignment
            vertically_aligned = abs(word_mid_y - widget_mid_y) <= VERTICAL_TOLERANCE

            # Check if word is to the right but not too far
            horizontally_close = (x0 > widget_rect.x1) and (
                x0 - widget_rect.x1 <= MAX_HORIZONTAL_DISTANCE
            )

            if vertically_aligned and horizontally_close:
                candidate_words.append((x0, word_text))

        if self.debug:
            self.logger.debug(
                "Label candidates | count=%d (vertical_tol=%d, max_dx=%d)",
                len(candidate_words),
                VERTICAL_TOLERANCE,
                MAX_HORIZONTAL_DISTANCE,
            )

        if not candidate_words:
            return None

        # Sort by horizontal position
        candidate_words.sort(key=lambda x: x[0])

        # Stop collecting words if there's a large gap (indicating next column)
        label_words = []
        MAX_WORD_GAP = 50  # maximum gap between consecutive words in same label (increased to capture multi-word options)

        for i, (x_pos, word_text) in enumerate(candidate_words):
            if i == 0:
                label_words.append(word_text)
            else:
                prev_x = candidate_words[i - 1][0]
                gap = x_pos - prev_x

                # If gap is too large, we've likely moved to next column
                if gap > MAX_WORD_GAP:
                    break
                label_words.append(word_text)

        label = " ".join(label_words)

        # Do NOT enhance at widget stage; enhancement is applied later with context
        if self.debug:
            self.logger.debug(
                "Resolved label='%s' (no enhancement at widget stage)", label
            )
        return label

    def _enhance_label_with_mappings(
        self, label: str, resource_key: str | None = None
    ) -> str:
        """
        Enhances extracted labels using predefined mappings to fix incomplete or truncated text.

        Args:
            label: The raw extracted label text
            resource_key: Optional mappings category key to restrict matches (e.g., 'NLP task type')

        Returns:
            Enhanced label text if mapping found, otherwise original label
        """
        if not label:
            return label

        # Sanitize noisy parts like "(e.g., ...)" from labels before comparing
        clean_label = self._sanitize_label_for_mapping(label)

        # If no mappings are configured, still return the sanitized label
        if not self.resource_mappings:
            return clean_label

        # Build iterable of categories to search (restricted if resource_key provided)
        categories_to_search = (
            [(resource_key, self.resource_mappings.get(resource_key, {}))]
            if resource_key
            else list(self.resource_mappings.items())
        )

        # Try to find a matching mapping key for this label within allowed categories
        for mapping_category, mappings in categories_to_search:
            # Direct match (case-insensitive)
            for mapped_label in mappings.keys():
                if clean_label.lower() == mapped_label.lower():
                    if self.debug:
                        self.logger.debug(
                            "Found direct mapping for '%s' -> '%s' in category '%s'",
                            clean_label,
                            mapped_label,
                            mapping_category,
                        )
                    return mapped_label

            # Partial match - look for labels that start with our extracted text (case-insensitive)
            for mapped_label in mappings.keys():
                if (
                    mapped_label.lower().startswith(clean_label.lower())
                    and len(clean_label) > 3
                ):
                    if self.debug:
                        self.logger.debug(
                            "Found partial mapping '%s' -> '%s' in category '%s'",
                            clean_label,
                            mapped_label,
                            mapping_category,
                        )
                    return mapped_label

            # Reverse partial match - check if our extracted text starts with a mapped label
            for mapped_label in mappings.keys():
                if (
                    clean_label.lower().startswith(mapped_label.lower())
                    and len(mapped_label) > 5
                ):
                    if self.debug:
                        self.logger.debug(
                            "Found reverse partial mapping '%s' -> '%s' in category '%s'",
                            clean_label,
                            mapped_label,
                            mapping_category,
                        )
                    return mapped_label

            # Fuzzy match for common truncation patterns (case-insensitive)
            for mapped_label in mappings.keys():
                # Check if our label is a truncated version of a mapped label
                if (
                    clean_label.lower() in mapped_label.lower()
                    and len(clean_label) > 5
                    and abs(len(mapped_label) - len(clean_label)) < 20
                ):
                    if self.debug:
                        self.logger.debug(
                            "Found fuzzy mapping '%s' -> '%s' in category '%s'",
                            clean_label,
                            mapped_label,
                            mapping_category,
                        )
                    return mapped_label

        # If no mapping found, return sanitized label (without e.g./i.e parentheses)
        return clean_label

    def _sanitize_label_for_mapping(self, label: str) -> str:
        """
        Removes parenthetical clarifications that include tokens like e.g., i.e., i.g. to
        make label matching against resource mappings more reliable.

        Examples:
            "Information extraction (e.g., features, terms) from" -> "Information extraction from"
        """
        if not label:
            return label

        text = label
        # Remove any parenthetical group that contains e.g., i.e., or i.g. (case-insensitive)
        # This targets only parentheses that include those markers to avoid deleting meaningful parts
        pattern = r"\s*\((?=[^)]*(?:e\.g\.|i\.e\.|i\.g\.))[^)]*\)"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove stray double spaces introduced by removal
        text = re.sub(r"\s+", " ", text).strip()

        # Remove any space before punctuation, and fix spaces around commas
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r",\s+", ", ", text)

        return text

    def _normalize_option_key(self, label: str) -> str:
        """
        Normalizes option labels for de-duplication:
        - lowercases
        - collapses all whitespace
        - trims punctuation spacing variants
        - removes parenthetical e.g./i.e./i.g. clarifications
        """
        if not label:
            return ""
        text = self._sanitize_label_for_mapping(label)
        # Normalize slashes: collapse spaces around '/'
        text = re.sub(r"\s*/\s*", " / ", text)
        # Lowercase
        text = text.lower()
        # Normalize common punctuation spacing
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r",\s+", ", ", text)
        # Collapse all whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_field_value(self, value) -> str:
        """
        Normalizes widget field_value:
        - returns empty string for None/"Off" (case-insensitive)
        - strips numeric-only tokens, including comma-separated like "Off, 9"
        - trims whitespace
        """
        if value is None:
            return ""
        text = str(value)
        # Split by commas, remove tokens that are 'off' or purely numeric
        parts = [p.strip() for p in text.split(",")]
        cleaned_parts = []
        for p in parts:
            if not p:
                continue
            if p.lower() == "off":
                continue
            if re.fullmatch(r"\d+", p):
                continue
            cleaned_parts.append(p)
        return ", ".join(cleaned_parts)

    def _iter_predicates(self):
        """
        Yields predicate info dicts from top-level and nested subtemplate_properties.
        """

        def _walk(node):
            if not isinstance(node, dict):
                return
            # If this looks like a predicate info (has at least a label/description), yield it
            if "label" in node and "description" in node:
                yield node
            # Recurse into nested subtemplate_properties
            subprops = (
                node.get("subtemplate_properties") if isinstance(node, dict) else None
            )
            if isinstance(subprops, dict):
                for _k, v in subprops.items():
                    yield from _walk(v)

        for _k, v in (self.predicates_mapping or {}).items():
            yield from _walk(v)

    def _build_question_mapping_index(self) -> dict:
        """
        Build a mapping from question_mapping token → resource_mapping_key.
        Handles both string and list forms of question_mapping.
        """
        index = {}
        for predicate_info in self._iter_predicates():
            resource_key = predicate_info.get("resource_mapping_key")
            if not resource_key:
                continue
            qmap = predicate_info.get("question_mapping")
            if not qmap:
                continue
            if isinstance(qmap, list):
                for token in qmap:
                    if isinstance(token, str):
                        index[token] = resource_key
            elif isinstance(qmap, str):
                index[qmap] = resource_key
        return index

    def _resolve_resource_key_for_question(self, question_text: str) -> str | None:
        """
        Determines the appropriate resource mapping category key for a given question text.

        Strategy:
        - Prefer matching by `question_mapping` token (e.g., 'II.1') parsed from question_text
        - Fallback to matching any known token appearing in the text
        - Fallback to matching predicate description substring across all predicate levels
        - Return the associated `resource_mapping_key` if found
        """
        if not question_text:
            return None

        # First pass: extract leading token like 'II.1'
        m = re.match(r"^\s*([IVXLCDM]+\.[0-9]+)", question_text)
        if m:
            token = m.group(1)
            rkey = self._question_mapping_index.get(token)
            if rkey and rkey in (self.resource_mappings or {}):
                return rkey

        # Second pass: any known token contained in question_text
        for token, rkey in self._question_mapping_index.items():
            if token in question_text and rkey in (self.resource_mappings or {}):
                return rkey

        # Third pass: match by description text across all predicates
        lowered_q = question_text.lower()
        for predicate_info in self._iter_predicates():
            desc = (predicate_info or {}).get("description", "").lower()
            if desc and (desc in lowered_q or (desc[:25] and desc[:25] in lowered_q)):
                rkey = predicate_info.get("resource_mapping_key")
                if rkey in (self.resource_mappings or {}):
                    return rkey

        return None

    def _get_expected_options_for_question(
        self, question_text: str, resource_key: str | None = None
    ) -> list:
        """
        Gets expected options for a question based on mappings.

        Args:
            question_text: The question text to find mappings for

        Returns:
            List of expected option labels
        """
        if not question_text or not self.predicates_mapping:
            return []

        # If resource_key already resolved, use it directly
        if resource_key and resource_key in self.resource_mappings:
            options = list(self.resource_mappings[resource_key].keys())
            if self.debug:
                self.logger.debug(
                    "Found %d expected options for question '%s' via key '%s'",
                    len(options),
                    question_text,
                    resource_key,
                )
            return options

        # Try resolving by leading token like 'II.1'
        m = re.match(r"^\s*([IVXLCDM]+\.[0-9]+)", question_text)
        if m:
            token = m.group(1)
            rkey = self._question_mapping_index.get(token)
            if rkey and rkey in self.resource_mappings:
                options = list(self.resource_mappings[rkey].keys())
                if self.debug:
                    self.logger.debug(
                        "Found %d expected options for question '%s' via token '%s'",
                        len(options),
                        question_text,
                        token,
                    )
                return options

        # Fallback: Try to match question text to predicate mappings by description across all levels
        lowered_q = question_text.lower()
        for predicate_info in self._iter_predicates():
            desc = (predicate_info.get("description", "") or "").lower()
            if desc and desc in lowered_q:
                r_key = predicate_info.get("resource_mapping_key")
                if r_key and r_key in self.resource_mappings:
                    options = list(self.resource_mappings[r_key].keys())
                    if self.debug:
                        self.logger.debug(
                            "Found %d expected options for question '%s' via predicate description",
                            len(options),
                            question_text,
                        )
                    return options

        return []

    def _validate_against_mappings(self, structured_data: dict) -> dict:
        """
        Validates extracted data against mappings and logs potential issues.

        Args:
            structured_data: The structured form data

        Returns:
            The validated structured data (with potential enhancements)
        """
        if not structured_data.get("questions"):
            return structured_data

        validation_summary = {
            "mapping_enhancements": 0,
            "potential_issues": [],
            "missing_options": 0,
        }

        for question in structured_data["questions"]:
            question_text = question.get("question_text", "")
            question_type = question.get("type", "")

            # Skip validation for text questions
            if question_type == "Text":
                continue

            # Check if we have expected options for this question
            expected_options = self._get_expected_options_for_question(question_text)
            if expected_options:
                found_options = set(question.get("all_options", []))
                missing_options = set(expected_options) - found_options

                if missing_options:
                    validation_summary["missing_options"] += len(missing_options)
                    if self.debug:
                        self.logger.debug(
                            "Question '%s' missing %d expected options: %s",
                            question_text[:50],
                            len(missing_options),
                            list(missing_options)[:3],  # Show first 3
                        )

            # Check for potential label enhancements using contextual resource key
            resource_key = self._resolve_resource_key_for_question(question_text)
            for option in question.get("options_details", []):
                original_label = option.get("label", "")
                if original_label:
                    enhanced = self._enhance_label_with_mappings(
                        original_label, resource_key
                    )
                    if enhanced != original_label:
                        validation_summary["mapping_enhancements"] += 1
                        if self.debug:
                            self.logger.debug(
                                "Enhanced option label: '%s' -> '%s'",
                                original_label,
                                enhanced,
                            )

        # Add validation summary to results
        if (
            validation_summary["mapping_enhancements"] > 0
            or validation_summary["missing_options"] > 0
        ):
            structured_data["validation_summary"] = validation_summary
            if self.debug:
                self.logger.info(
                    "Validation complete: %d enhancements, %d missing options",
                    validation_summary["mapping_enhancements"],
                    validation_summary["missing_options"],
                )

        return structured_data

    def _merge_duplicate_questions(self, structured_data: dict) -> dict:
        """
        Post-processes the structured data to merge duplicate questions with the same question_text.
        When a question appears as both a choice-type (RadioButton/CheckBox) and a text field,
        appends the text field answer to the selected_answers of the choice-type question.
        Also injects a synthetic option into options_details with source_type="Text" so the
        origin of the merged answer is preserved.
        """
        questions = structured_data.get("questions", [])
        if not questions:
            return structured_data

        # Group questions by question_text
        question_groups = {}
        for question in questions:
            question_text = question.get("question_text", "")
            if question_text not in question_groups:
                question_groups[question_text] = []
            question_groups[question_text].append(question)

        # Process groups with multiple questions (duplicates)
        merged_questions = []
        for question_text, question_list in question_groups.items():
            if len(question_list) == 1:
                # No duplicates, keep as is
                merged_questions.append(question_list[0])
            else:
                # Found duplicates, merge them
                if self.debug:
                    self.logger.debug(
                        "Merging duplicate questions | text='%s' count=%d",
                        question_text,
                        len(question_list),
                    )
                merged_question = self._merge_question_group(question_list)
                merged_questions.append(merged_question)

        # Update the structured data with merged questions
        structured_data["questions"] = merged_questions
        structured_data["total_questions"] = len(merged_questions)

        # Recalculate questions_with_selections
        def question_has_answer(q: dict) -> bool:
            qtype = q.get("type")
            if qtype == "Text":
                return bool(q.get("answer"))
            # Choice types
            selected = q.get("selected_answers")
            if selected is not None:
                return any(ans and ans != "None" for ans in selected)
            # Fallback using options_details
            for opt in q.get("options_details", []) or []:
                if opt.get("is_selected"):
                    return True
            return False

        questions_with_selections = sum(
            1 for q in merged_questions if question_has_answer(q)
        )
        structured_data["extraction_summary"][
            "questions_with_selections"
        ] = questions_with_selections

        return structured_data

    def _merge_question_group(self, question_list: list) -> dict:
        """
        Merges a group of questions with the same question_text.
        Prioritizes choice-type questions (RadioButton/CheckBox) and appends text field answers.
        When merging a text field answer, additionally adds it as an option with
        source_type="Text" to options_details (and to all_options) and marks it selected.
        """
        # Find the choice-type question (RadioButton/CheckBox) and text field question
        choice_question = None
        text_question = None

        for question in question_list:
            question_type = question.get("type", "")
            if question_type in ["RadioButton", "CheckBox"]:
                choice_question = question
            elif question_type == "Text":
                text_question = question

        # If we have both choice and text questions, merge them
        if choice_question and text_question:
            # Get the text answer
            text_answer = text_question.get("answer", "").strip()

            # Handle text answer for selected_answers
            if text_answer:
                selected_answers = choice_question.get("selected_answers", [])

                # Check if "Other/Comments" is in the selected answers
                other_comments_found = any(
                    any(
                        other_comment.lower() in answer.lower()
                        for other_comment in self.list_of_other_comments
                    )
                    for answer in selected_answers
                )

                if other_comments_found:
                    # Replace "Other/Comments" with the actual comment text
                    new_selected_answers = []
                    for answer in selected_answers:
                        is_other_comment = any(
                            other_comment.lower() in answer.lower()
                            for other_comment in self.list_of_other_comments
                        )
                        if is_other_comment:
                            new_selected_answers.append(text_answer)
                        else:
                            new_selected_answers.append(answer)
                    selected_answers = new_selected_answers
                elif selected_answers and selected_answers != ["None"]:
                    # Append the text answer to existing selected answers
                    selected_answers.append(text_answer)
                else:
                    # If no other selections, just use the text answer
                    selected_answers = [text_answer]
                choice_question["selected_answers"] = selected_answers

                # Ensure the merged text also appears as an option with provenance
                # 1) Add to all_options if not already present
                all_options = choice_question.get("all_options") or []
                if text_answer not in all_options:
                    all_options.append(text_answer)
                    choice_question["all_options"] = all_options

                # 2) Add to options_details with source_type indicating it came from Text
                options_details = choice_question.get("options_details") or []
                # Check if an option with the same label already exists
                existing_opt = next(
                    (o for o in options_details if o.get("label") == text_answer), None
                )
                if existing_opt:
                    # Mark as selected and keep any existing fields; annotate source_type if missing
                    existing_opt["is_selected"] = True
                    if not existing_opt.get("source_type"):
                        existing_opt["source_type"] = "Text"
                    if not existing_opt.get("field_value"):
                        existing_opt["field_value"] = text_answer
                else:
                    options_details.append(
                        {
                            "label": text_answer,
                            "field_name": text_question.get("field_name"),
                            "field_value": text_answer,
                            "is_selected": True,
                            "source_type": "Text",
                        }
                    )
                choice_question["options_details"] = options_details

                # 3) Update total_options to reflect any addition
                choice_question["total_options"] = len(
                    choice_question.get("options_details") or []
                )
                if self.debug:
                    self.logger.debug(
                        "Merged text answer into choices | text='%s' -> selected=%s (as option with source_type=Text)",
                        text_answer,
                        ", ".join(choice_question.get("selected_answers", []))
                        or "None",
                    )

            return choice_question
        else:
            # If only one type exists, return the first one
            return question_list[0]

    def to_json(self, indent: int = 2) -> str:
        """
        Converts the extracted results dictionary to a JSON formatted string.
        """
        return json.dumps(self.results, indent=indent, ensure_ascii=False)
