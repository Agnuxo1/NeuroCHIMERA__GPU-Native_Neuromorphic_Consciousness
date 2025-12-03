# Instructions to Generate PDF from HTML Paper

## Method 1: Using Web Browser (RECOMMENDED for Windows)

### Google Chrome / Microsoft Edge (Best Quality)

1. Open the HTML file in Chrome or Edge:
   - Double-click `NeuroCHIMERA_Paper.html`
   - Or File → Open → Select the HTML file

2. Press `Ctrl+P` (or File → Print)

3. Configure print settings:
   - **Destination:** Save as PDF
   - **Layout:** Portrait
   - **Paper size:** A4
   - **Margins:** Default (or Custom: 2cm all sides)
   - **Options:** Enable "Background graphics"
   - **Scale:** 100%

4. Click "Save" and choose destination

5. The PDF will maintain the 2-column format perfectly

### Firefox

1. Open HTML file in Firefox
2. Press `Ctrl+P`
3. Select "Save to PDF"
4. Configure:
   - Paper size: A4
   - Orientation: Portrait
   - Print backgrounds: ON
5. Click "Save"

## Method 2: Using wkhtmltopdf (Alternative)

If you need command-line generation:

```bash
# Download wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html
# Install it, then run:

wkhtmltopdf --page-size A4 --margin-top 20mm --margin-bottom 20mm --margin-left 20mm --margin-right 20mm --enable-local-file-access NeuroCHIMERA_Paper.html NeuroCHIMERA_Paper.pdf
```

## Method 3: Using Python with Playwright (Advanced)

```python
# Install: pip install playwright
# Then: playwright install chromium

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('file:///d:/Vladimir/NeuroCHIMERA_Paper.html')
    page.pdf(
        path='NeuroCHIMERA_Paper.pdf',
        format='A4',
        print_background=True,
        margin={'top': '20mm', 'right': '20mm', 'bottom': '20mm', 'left': '20mm'}
    )
    browser.close()
```

## Method 4: Online Tools

1. Upload HTML to:
   - https://www.sejda.com/html-to-pdf
   - https://cloudconvert.com/html-to-pdf
   - https://www.sodapdf.com/html-to-pdf/

2. Download generated PDF

## Quality Verification

After generating PDF, verify:
- ✓ 2-column format is preserved
- ✓ SVG figures render correctly
- ✓ Tables have proper formatting
- ✓ Equations are readable
- ✓ References are numbered correctly
- ✓ Page breaks don't split figures/tables
- ✓ All colors and gradients are visible
- ✓ Links are clickable (test GitHub, ResearchGate links)

## Recommended Settings Summary

- **Format:** A4 (210 × 297 mm)
- **Orientation:** Portrait
- **Margins:** 20mm all sides
- **Background Graphics:** ENABLED (critical for SVG figures)
- **Scale:** 100%
- **Headers/Footers:** None

## File Information

- **Source HTML:** `d:\Vladimir\NeuroCHIMERA_Paper.html`
- **Expected PDF:** `d:\Vladimir\NeuroCHIMERA_Paper.pdf`
- **Word Count:** ~15,000 words
- **Figures:** 3 professional SVG figures
- **Tables:** 6 comprehensive tables
- **References:** 45 citations with DOIs
- **Pages:** Expected 25-30 pages in 2-column format

## Authors

- **V.F. Veselov** - Theoretical Framework & HNS (Moscow Institute of Electronic Technology)
- **Francisco Angulo de Lafuente** - CHIMERA Implementation (Madrid, Spain)

## Note on WeasyPrint

WeasyPrint has dependency issues on Windows (requires GTK libraries). The browser method is simpler and produces equivalent or better results. If you're on Linux:

```bash
pip install weasyprint
python -c "from weasyprint import HTML; HTML(filename='NeuroCHIMERA_Paper.html').write_pdf('NeuroCHIMERA_Paper.pdf')"
```
