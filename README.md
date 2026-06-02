# NLP4RE ID Card Processing System
<div align="center">

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
[![Paper](https://img.shields.io/badge/CEUR--WS-Vol--4208-blue)](https://ceur-ws.org/Vol-4208/nlp4re-short4.pdf)

</div>

This repository contains the implementation of an automated system for extracting and processing NLP4RE (Natural Language Processing for Requirements Engineering) ID Card data from PDF forms. The system converts interactive PDF form submissions into structured JSON format suitable for integration with the Open Research Knowledge Graph (ORKG).

## Overview

The system addresses the challenge of extracting structured data from interactive PDF forms used in research surveys. It employs spatial analysis techniques to identify form field labels and implements intelligent text processing to handle various field types including checkboxes, radio buttons, and text inputs.

## Key Components

### PDFFormExtractor
The core extraction module (`scripts/PDFFormExtractor.py`) implements advanced PDF processing techniques including:
- Spatial analysis for automatic label detection
- Support for multiple field types (CheckBox, RadioButton, Text)
- Intelligent handling of "Other/Comments" fields with text content extraction
- Integration with predefined mappings for data validation and enhancement

### Data Processing Pipeline
- `pdf2JSON.py` - Single PDF extraction
- `batch_process.py` - Batch processing for multiple files
- `create_instance.py` - ORKG template generation

## Usage

```bash
# Extract data from a single PDF
python pdf2JSON.py path/to/form.pdf

# Process multiple PDFs
python batch_process.py
```

### ORKG Configuration

The ORKG connection settings are configured in `scripts/config.py`:

- **`ORKG_HOST`**: Base URL of the ORKG instance to connect to (default: `https://orkg.org`).
- **`ORKG_USERNAME` / `ORKG_PASSWORD`**: ORKG account credentials used when creating templates or instances. **Do not commit real credentials**; either set them via environment variables and load them in `config.py`, or keep this file local and excluded from version control.

Set these values to match your ORKG deployment before running `create_instance.py` or any scripts that interact with ORKG.

## Technical Implementation

The system utilizes PyMuPDF for PDF processing and implements custom algorithms for:
- Form field detection and classification
- Label extraction using spatial proximity analysis
- Text content preservation from comment fields
- Data validation against predefined mapping schemas

## Requirements

- Python 3.7+
- PyMuPDF
- See `requirements.txt` for complete dependencies

## Repository Structure

```
├── scripts/           # Core processing modules
├── pdf_files/         # Source PDF files and extracted data
├── pdf2JSON_Results/  # Additional processing results
├── template_info/     # ORKG template specifications
└── run_logs/         # Processing logs
```

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.
