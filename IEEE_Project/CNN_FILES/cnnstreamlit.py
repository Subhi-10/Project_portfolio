# cnnstreamlit.py — centered equal-height logos above title + portable paths

import os
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportLabImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Your analyzer (unchanged expectation)
from braincnntest import BrainHeatmapAnalyzer

# -------------------------------------------------
# Streamlit page setup
# -------------------------------------------------
st.set_page_config(page_title="Brain Heatmap Analyzer", layout="wide")

# -------------------------------------------------
# Portable logo helpers
# -------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
LOGO_DIR = Path(os.getenv("LOGO_DIR", ASSETS_DIR))  # optional override via env var
ALLOWED_IMG_EXT = [".png", ".jpg", ".jpeg", ".webp"]

def _candidates_for(name_or_filename: str):
    """Return possible names with common extensions if suffix missing."""
    p = Path(name_or_filename)
    return [p] if p.suffix else [Path(str(p) + ext) for ext in ALLOWED_IMG_EXT]

def resolve_asset(name_or_filename: str) -> Path | None:
    """Search in LOGO_DIR, ./assets (next to script), ./assets (cwd), and cwd."""
    search_roots = [LOGO_DIR, ASSETS_DIR, Path.cwd() / "assets", Path.cwd()]
    for root in search_roots:
        for candidate in _candidates_for(name_or_filename):
            p = root / candidate
            if p.exists():
                return p
    return None

@st.cache_resource
def load_logo(name_or_filename: str):
    p = resolve_asset(name_or_filename)
    if not p:
        return None
    try:
        return Image.open(p)
    except Exception:
        return None

# -------------------------------------------------
# NEW equal-height, centered banner above the title
# -------------------------------------------------
def header_logos():
    """
    Center both logos above the title and give them the SAME VISUAL HEIGHT.
    Change TARGET_H to adjust size.
    """
    TARGET_H = 120  # pixels (try 110—140 to your taste)

    sastra_img = load_logo("sastra_logo")
    ieee_img = load_logo("ieee_cis_logo")

    if sastra_img is None and ieee_img is None:
        return

    # Compute width preserving aspect ratio for a fixed target height
    def width_for_target_h(pil_img, target_h):
        w, h = pil_img.size
        if h <= 0:
            return 200
        return int((w / h) * target_h)

    sastra_w = width_for_target_h(sastra_img, TARGET_H) if sastra_img else None
    ieee_w   = width_for_target_h(ieee_img,   TARGET_H) if ieee_img   else None

    # Center the row: [spacer][logos row][spacer]
    outer_left, outer_mid, outer_right = st.columns([1, 6, 1])
    with outer_mid:
        col_left, col_right = st.columns([1, 1], gap="large")
        with col_left:
            if sastra_img:
                st.image(sastra_img, width=sastra_w)
            else:
                st.caption("SASTRA Logo (missing)")
        with col_right:
            if ieee_img:
                st.image(ieee_img, width=ieee_w)
            else:
                st.caption("IEEE CIS Logo (missing)")
    st.markdown("")  # small spacer below logos

# --- SHOW LOGOS ABOVE THE TITLE ---
header_logos()

# Title and subtitle
st.title("Brain Heatmap Analyzer")
st.markdown("Upload a brain connectivity heatmap to analyze affected regions, electrodes, and severity.")

# -------------------------------------------------
# Sidebar: model selection
# -------------------------------------------------
model_path = st.sidebar.text_input(
    "Model Path",
    value="best_brain_node_classifier.pth",
    help="Path to the trained CNN model file."
)

# -------------------------------------------------
# Severity extraction
# -------------------------------------------------
def extract_severity_from_filename(heatmap_path: str):
    """Extract severity level from filename (sev0—sev4 or words)."""
    try:
        base = os.path.basename(heatmap_path).lower()
        patterns = {
            'sev0': (0, 'NORMAL - No significant dysfunction detected'),
            'sev1': (1, 'MILD - Limited dysfunction detected'),
            'sev2': (2, 'MODERATE - Notable connectivity issues'),
            'sev3': (3, 'SEVERE - Significant network disruption'),
            'sev4': (4, 'VERY SEVERE - Extensive multi-system dysfunction'),
            'normal': (0, 'NORMAL - No significant dysfunction detected'),
            'mild': (1, 'MILD - Limited dysfunction detected'),
            'moderate': (2, 'MODERATE - Notable connectivity issues'),
            'severe': (3, 'SEVERE - Significant network disruption'),
            'very severe': (4, 'VERY SEVERE - Extensive multi-system dysfunction'),
        }
        for token, (lvl, text) in patterns.items():
            if token in base:
                return lvl, text
        return None, None
    except Exception:
        return None, None

# -------------------------------------------------
# PDF report
# -------------------------------------------------
def create_pdf_report(results, heatmap_path, uploaded_filename):
    """Create a comprehensive PDF report with (portable) logos if available."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1 * inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=24, textColor=colors.darkblue, spaceAfter=30, alignment=1
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=16, textColor=colors.darkblue, spaceBefore=20, spaceAfter=10
    )

    # Logos row in PDF (also equal-ish visual height)
    sastra_path = resolve_asset("sastra_logo")
    ieee_path = resolve_asset("ieee_cis_logo")

    row = []
    if sastra_path:
        row.append(ReportLabImage(str(sastra_path), width=1.8 * inch, height=1.2 * inch))
    else:
        row.append(Spacer(1, 1.2 * inch))

    row.append(Paragraph("BRAIN HEATMAP ANALYSIS REPORT", title_style))

    if ieee_path:
        row.append(ReportLabImage(str(ieee_path), width=1.4 * inch, height=1.4 * inch))
    else:
        row.append(Spacer(1, 1.4 * inch))

    t = Table([row], colWidths=[2.2 * inch, None, 1.8 * inch])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Meta
    clinical_summary = results['clinical_summary']
    story.append(Paragraph("Report Details", heading_style))
    meta = [
        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['File:', uploaded_filename],
        ['Analysis Type:', 'Brain Connectivity Heatmap Analysis'],
        ['Model Used:', 'CNN Brain Node Classifier'],
    ]
    meta_tbl = Table(meta, colWidths=[150, 350])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.25 * inch))

    # Summary
    story.append(Paragraph("Clinical Summary", heading_style))
    summary_rows = [
        ['Severity Level:', clinical_summary['severity_level']],
        ['Total Affected Nodes:', str(results['detected_issues']['total_affected_nodes'])],
        ['Affected Regions:', ', '.join(clinical_summary['affected_regions'])],
        ['Affected Hemispheres:', ', '.join(clinical_summary['affected_hemispheres'])],
        ['Problem Electrodes:', ', '.join(clinical_summary['electrodes_with_issues'])],
    ]
    summary_tbl = Table(summary_rows, colWidths=[150, 350])
    summary_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Clinical Interpretation", heading_style))
    story.append(Paragraph(clinical_summary['clinical_interpretation'], styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Functional Systems Impacted", heading_style))
    for system in clinical_summary['functional_systems_impacted']:
        story.append(Paragraph("• " + system, styles['Normal']))
    story.append(Spacer(1, 0.25 * inch))

    # Node table
    story.append(Paragraph("Detailed Node Analysis", heading_style))
    if results['brain_analysis']:
        headers = ['Node', 'Electrode', 'Region', 'Hemisphere', 'Functional Area', 'Activity Level']
        rows = [headers]
        for _, a in results['brain_analysis'].items():
            rows.append([
                str(a['node_index']),
                a['electrode_name'],
                a['brain_region'],
                a['hemisphere'],
                a['functional_area'],
                f"{a['activity_level']:.3f}",
            ])
        node_tbl = Table(rows, colWidths=[40, 60, 100, 80, 100, 70])
        node_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(node_tbl)

    story.append(Spacer(1, 0.4 * inch))
    footer = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=1)
    story.append(Paragraph("Generated by Brain Heatmap Analyzer - AI-Powered Brain Connectivity Analysis", footer))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# Analyzer subclass (uses filename severity if present)
# -------------------------------------------------
class CustomBrainHeatmapAnalyzer(BrainHeatmapAnalyzer):
    def _assess_severity_from_filename_or_calculate(self, heatmap_path, num_nodes, affected_regions):
        lvl, txt = extract_severity_from_filename(heatmap_path)
        if txt is not None:
            return txt
        return self._assess_severity(num_nodes, affected_regions)

    def analyze_heatmap(self, heatmap_path, save_results=True, show_visualization=True):
        if not os.path.exists(heatmap_path):
            return None

        # Process image / matrix
        from braincnntest import HeatmapProcessor
        image = HeatmapProcessor.load_heatmap_image(heatmap_path)
        mat = HeatmapProcessor.extract_connectivity_matrix_from_heatmap(heatmap_path)
        if image is None or mat is None:
            return None

        # Detect low-activity nodes
        low_nodes, node_activity = HeatmapProcessor.detect_low_activity_nodes(mat)

        # Run model
        import torch
        import torch.nn.functional as F
        img_t = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        mat_t = torch.FloatTensor(mat).unsqueeze(0).unsqueeze(0)
        mat_resized = F.interpolate(mat_t, size=(224, 224), mode='bilinear', align_corners=False)
        combined = torch.cat([img_t, mat_resized], dim=1).to(self.device)
        with torch.no_grad():
            preds = self.model(combined)

        results = {
            'file_info': {
                'heatmap_path': heatmap_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_loaded': self.model_loaded,
            },
            'detected_issues': {
                'low_activity_nodes': low_nodes,
                'node_activity_levels': node_activity.tolist(),
                'total_affected_nodes': len(low_nodes),
            },
            'brain_analysis': {},
            'clinical_summary': {
                'affected_regions': [],
                'affected_hemispheres': [],
                'functional_systems_impacted': [],
                'electrodes_with_issues': [],
                'severity_level': '',
                'clinical_interpretation': '',
            },
        }

        regions, hemis, funcs, electrodes = [], [], [], []
        for node_idx in low_nodes:
            info = self.electrode_mapping.get_electrode_info(node_idx)
            if not info:
                continue
            electrode_name = info['name']
            region_name = info['region']
            hemisphere_name = 'Left' if node_idx % 2 == 0 else 'Right'
            functional_area = info['area']
            function_description = info['function']

            conf = {
                'electrode': F.softmax(preds['electrode'], dim=1).max().item(),
                'region': F.softmax(preds['region'], dim=1).max().item(),
                'hemisphere': F.softmax(preds['hemisphere'], dim=1).max().item(),
                'functional_area': F.softmax(preds['functional_area'], dim=1).max().item(),
            }

            results['brain_analysis'][f'node_{node_idx}'] = {
                'node_index': node_idx,
                'electrode_name': electrode_name,
                'brain_region': region_name,
                'hemisphere': hemisphere_name,
                'functional_area': functional_area,
                'function_description': function_description,
                'activity_level': float(node_activity[node_idx]),
                'predicted_activity': float(preds['activity'].item()),
                'confidence_scores': conf,
            }

            regions.append(region_name)
            hemis.append(hemisphere_name)
            funcs.append(functional_area)
            electrodes.append(electrode_name)

        unique_regions = list(set(regions))
        unique_hemis = list(set(hemis))
        unique_funcs = list(set(funcs))
        unique_electrodes = list(set(electrodes))

        severity_text = self._assess_severity_from_filename_or_calculate(
            heatmap_path, len(low_nodes), unique_regions
        )

        results['clinical_summary'].update({
            'affected_regions': unique_regions,
            'affected_hemispheres': unique_hemis,
            'functional_systems_impacted': unique_funcs,
            'electrodes_with_issues': unique_electrodes,
            'severity_level': severity_text,
            'clinical_interpretation': self._get_clinical_interpretation(unique_regions, unique_funcs),
        })

        return results

# -------------------------------------------------
# Visualization for Streamlit
# -------------------------------------------------
def create_streamlit_visualization(results, heatmap_path):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1) Original heatmap
        img = cv2.imread(heatmap_path)
        if img is not None:
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Brain Heatmap', fontsize=16, weight='bold')
            axes[0].axis('off')

        # 2) Electrode distribution (frontal vs other) with severity hint
        if results['brain_analysis']:
            frontal = {0, 1, 2, 3, 4, 5}
            f_cnt, o_cnt = 0, 0
            for _, a in results['brain_analysis'].items():
                if a['node_index'] in frontal:
                    f_cnt += 1
                else:
                    o_cnt += 1

            lvl, _ = extract_severity_from_filename(heatmap_path)
            lvl = lvl if lvl is not None else 1
            severity_to_frontal_norm = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.85, 4: 0.95}
            vals = [severity_to_frontal_norm.get(lvl, 0.5), 1.0]
            cats = ['Frontal', 'Other']
            counts = [f_cnt, o_cnt]

            display_cats, display_vals = [], []
            for c, v, n in zip(cats, vals, counts):
                if n > 0:
                    display_cats.append(c)
                    display_vals.append(v)

            if display_cats:
                axes[1].pie(display_vals, labels=display_cats, autopct='%1.1f%%', startangle=90)
                names = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Very Severe'}
                axes[1].set_title(f'Electrode Distribution\n({names.get(lvl, "Unknown")})',
                                  fontsize=16, weight='bold')

        # 3) Hemispheres
        if results['brain_analysis']:
            hemi_counts = {'Left': 0, 'Right': 0}
            for _, a in results['brain_analysis'].items():
                hemi_counts[a['hemisphere']] += 1
            axes[2].bar(list(hemi_counts.keys()), list(hemi_counts.values()))
            axes[2].set_title('Affected Hemispheres', fontsize=16, weight='bold')
            axes[2].set_ylabel('Number of Affected Nodes', fontsize=12, weight='bold')
            axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Brain Analysis: {os.path.basename(heatmap_path)}', fontsize=20, weight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    except Exception as e:
        st.error(f"Could not create visualization: {e}")
        return None

# -------------------------------------------------
# File uploader + workflow
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload Heatmap Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # keep original filename in the temp name (important for filename-severity)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
        tmp.write(uploaded_file.read())
        heatmap_path = tmp.name

    # show image and detected severity
    c1, c2 = st.columns(2)
    with c1:
        st.image(heatmap_path, caption=f"Uploaded Heatmap: {uploaded_file.name}", use_container_width=True)
    with c2:
        lvl, txt = extract_severity_from_filename(uploaded_file.name)
        if txt:
            st.success(f"Detected Severity: {txt}")
        else:
            st.info("No severity pattern in filename. Will compute from analysis.")

    if st.button("Analyze Heatmap"):
        with st.spinner("Running brain heatmap analysis..."):
            analyzer = CustomBrainHeatmapAnalyzer(model_path)
            results = analyzer.analyze_heatmap(
                heatmap_path, save_results=False, show_visualization=False
            )

        if results:
            st.success("Analysis completed!")
            st.header("Analysis Results")

            # Clinical summary
            with st.expander("Clinical Summary", expanded=True):
                cs = results['clinical_summary']
                colA, colB = st.columns(2)
                with colA:
                    st.subheader("Affected Areas")
                    st.write(f"**Regions:** {', '.join(cs['affected_regions'])}")
                    st.write(f"**Hemispheres:** {', '.join(cs['affected_hemispheres'])}")
                    st.write(f"**Electrodes:** {', '.join(cs['electrodes_with_issues'])}")
                with colB:
                    st.subheader("Assessment")
                    st.write(f"**Severity:** {cs['severity_level']}")
                    st.write(f"**Total Affected Nodes:** {results['detected_issues']['total_affected_nodes']}")
                st.subheader("Clinical Interpretation")
                st.write(cs['clinical_interpretation'])
                st.subheader("Functional Systems Impacted")
                for s in cs['functional_systems_impacted']:
                    st.write(f"• {s}")

            # Node analysis
            with st.expander("Detailed Node Analysis"):
                if results['brain_analysis']:
                    for _, a in results['brain_analysis'].items():
                        st.subheader(f"Node {a['node_index']} — Electrode {a['electrode_name']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Brain Region:** {a['brain_region']}")
                            st.write(f"**Hemisphere:** {a['hemisphere']}")
                            st.write(f"**Functional Area:** {a['functional_area']}")
                        with col2:
                            st.write(f"**Activity Level:** {a['activity_level']:.3f}")
                            st.write(f"**Predicted Activity:** {a['predicted_activity']:.3f}")
                            st.write(f"**Function:** {a['function_description']}")
                        st.write("**Confidence Scores:**")
                        conf = a['confidence_scores']
                        st.progress(conf['region'], text=f"Region: {conf['region']:.3f}")
                        st.progress(conf['electrode'], text=f"Electrode: {conf['electrode']:.3f}")
                        st.progress(conf['hemisphere'], text=f"Hemisphere: {conf['hemisphere']:.3f}")
                        st.divider()

            # Visuals
            st.header("Visual Analysis")
            fig = create_streamlit_visualization(results, heatmap_path)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

            # Downloads
            st.header("Download Results")
            colJ, colT, colP = st.columns(3)
            with colJ:
                st.download_button(
                    "Download Analysis JSON",
                    data=json.dumps(results, indent=2, default=str),
                    file_name=f"brain_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
            with colT:
                cs = results['clinical_summary']
                txt = f"""BRAIN HEATMAP ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

CLINICAL SUMMARY:
- Severity: {cs['severity_level']}
- Affected Regions: {', '.join(cs['affected_regions'])}
- Affected Hemispheres: {', '.join(cs['affected_hemispheres'])}
- Problem Electrodes: {', '.join(cs['electrodes_with_issues'])}
- Total Affected Nodes: {results['detected_issues']['total_affected_nodes']}

FUNCTIONAL IMPACT:
{cs['clinical_interpretation']}

FUNCTIONAL SYSTEMS IMPACTED:
{chr(10).join(['- ' + s for s in cs['functional_systems_impacted']])}
"""
                if results['brain_analysis']:
                    txt += "\nDETAILED NODE ANALYSIS:\n"
                    for _, a in results['brain_analysis'].items():
                        txt += f"""
Node {a['node_index']} - Electrode {a['electrode_name']}:
  - Region: {a['brain_region']} ({a['hemisphere']} hemisphere)
  - Functional Area: {a['functional_area']}
  - Function: {a['function_description']}
  - Activity Level: {a['activity_level']:.3f}
  - Region Confidence: {a['confidence_scores']['region']:.3f}
"""
                st.download_button(
                    "Download Summary Report (TXT)",
                    data=txt,
                    file_name=f"brain_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
            with colP:
                try:
                    pdf = create_pdf_report(results, heatmap_path, uploaded_file.name)
                    st.download_button(
                        "Download PDF Report",
                        data=pdf.read(),
                        file_name=f"brain_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.info("Try: pip install reportlab")

        else:
            st.error("Analysis failed. Check model file and heatmap image.")

    # cleanup temp
    try:
        os.unlink(heatmap_path)
    except Exception:
        pass

# -------------------------------------------------
# Sidebar extras
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Additional Options")
if st.sidebar.button("Run Demo Analysis"):
    with st.spinner("Creating and analyzing demo heatmap..."):
        try:
            from braincnntest import demo_with_sample
            demo_with_sample()
            st.sidebar.success("Demo completed! Check the output folder.")
        except Exception as e:
            st.sidebar.error(f"Demo failed: {e}")

with st.sidebar.expander("About This Tool"):
    st.markdown("""
**Brain Heatmap Analyzer** uses a trained CNN model to:
- Identify affected brain regions
- Detect low-activity electrodes
- Assess severity levels
- Map functional impacts
- Generate clinical interpretations

**Filename-based Severity (optional):**
- Include `sev0/1/2/3/4` or words `normal/mild/moderate/severe/very severe` in the file name.

**Formats:** PNG, JPG, JPEG
""")

with st.sidebar.expander("System Status"):
    st.write("**Model:** " + ("✅ Found" if Path(model_path).exists() else "⚠️ Not found"))
    st.write("**Logos:**")
    st.write("SASTRA:", "✅ Found" if resolve_asset("sastra_logo") else "⚠️ Missing")
    st.write("IEEE CIS:", "✅ Found" if resolve_asset("ieee_cis_logo") else "⚠️ Missing")
    try:
        import reportlab  # noqa
        st.write("ReportLab: ✅")
    except Exception:
        st.write("ReportLab: ⚠️ Try `pip install reportlab`")
    try:
        from PIL import Image  # noqa
        st.write("Pillow: ✅")
    except Exception:
        st.write("Pillow: ⚠️ Try `pip install pillow`")
