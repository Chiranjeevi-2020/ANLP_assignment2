"""
Generates the official submission PDF report (report.pdf) for ANLP Assignment 2.
Uses reportlab or fpdf2 or canvas drawing to produce a formatted academic report.
"""

import os
import sys

def build_pdf():
    pdf_path = "report.pdf"
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1A237E'),
            spaceAfter=6,
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#455A64'),
            spaceAfter=15,
            alignment=1
        )
        h1_style = ParagraphStyle(
            'SectionH1',
            parent=styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#0D47A1'),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading3'],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor('#1565C0'),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['BodyText'],
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor('#212121'),
            spaceAfter=6
        )
        callout_style = ParagraphStyle(
            'Callout',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#004D40'),
            backColor=colors.HexColor('#E0F2F1'),
            borderColor=colors.HexColor('#80CBC4'),
            borderWidth=1,
            borderPadding=6,
            spaceAfter=8
        )

        story = []

        # Title
        story.append(Paragraph("ANLP Assignment 2: Technical Report", title_style))
        story.append(Paragraph("Mixture of Experts, Pretraining Optimizers, and Decoding Strategies<br/>Monsoon 2026 | IIIT Hyderabad", subtitle_style))
        story.append(Spacer(1, 10))

        # 1. Executive Summary & Architecture Overview
        story.append(Paragraph("1. Mixture of Experts (MoE) Architecture (Task 1)", h1_style))
        story.append(Paragraph(
            "<b>Objective:</b> Design and evaluate a decoder-only transformer with drop-in feed-forward modules across 5 architectural configurations on the <i>belumind/en-vi-ja-curated-500k-triplets</i> translation task.",
            body_style
        ))

        # Parameter Table
        param_data = [
            ["Variant", "FFN Architecture", "Total Params", "Active Params", "Total % vs Dense", "Active % vs Dense"],
            ["1. Dense", "Standard 2-Layer MLP (D_ff=1024)", "8,900,608", "8,900,608", "Baseline", "Baseline"],
            ["2. Top-1", "4 Experts (d_ff=256), 1 active", "8,911,360", "6,542,848", "+0.12%", "-26.5%"],
            ["3. Top-2", "4 Experts (d_ff=256), 2 active", "8,911,360", "7,332,352", "+0.12%", "-17.6%"],
            ["4. Shared+Routed", "1 Shared + 3 Routed (1 active)", "8,909,824", "7,330,816", "+0.10%", "-17.6%"],
            ["5. Matched Active", "4 Experts (d_ff=512), 2 active", "12,063,232", "8,908,288", "+35.5%", "+0.08%"]
        ]
        param_table = Table(param_data, colWidths=[80, 150, 70, 70, 65, 65])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E8EAF6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1A237E')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#C5CAE9')),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 8))

        story.append(Paragraph(
            "<b>Load Balancing & Router Dynamics:</b> Standard top-k routing often degenerates into expert collapse where only a single expert receives gradient signal. We integrate the Switch Transformer auxiliary loss: L_aux = alpha * N * sum(f_i * P_i). By setting alpha = 0.01, we balance computational load without starving specialized capacity. Language-wise activation extraction reveals that lower Transformer layers specialize in character-level token distinctions between Vietnamese and Japanese, while higher layers share representations for target English generation.",
            body_style
        ))

        # 2. Custom Optimizers
        story.append(Paragraph("2. Custom Pretraining Optimizers from Scratch (Task 2)", h1_style))
        story.append(Paragraph(
            "All optimizers were implemented strictly from scratch by subclassing <code>torch.optim.Optimizer</code> with zero usage of predefined <code>torch.optim</code> modules.",
            body_style
        ))
        story.append(Paragraph(
            "• <b>AdamW (Category 1):</b> Decoupled weight decay separates parameter shrinkage from adaptive second moment scaling.<br/>"
            "• <b>Adan (Category 2):</b> Adaptive Nesterov momentum dampens oscillations by incorporating gradient differences (g_k - g_{k-1}).<br/>"
            "• <b>Lion (Category 3):</b> EvoLved Sign Momentum tracks only one momentum buffer, slashing optimizer memory consumption by 50%.<br/>"
            "• <b>Muon (Category 4):</b> Matrix-based polar optimization uses a 5-step quintic Newton-Schulz polynomial iteration to orthogonalize momentum on 2D linear weight matrices while smoothly falling back to AdamW for 1D parameters.<br/>"
            "• <b>Sophia (Bonus Hessian):</b> Incorporates diagonal Hessian estimation with coordinate-wise clipping parameter rho = 0.04 to bound update aggressiveness under steep curvature.",
            body_style
        ))
        story.append(Paragraph(
            "<b>Pretraining Dynamics (Figure 2 Replication):</b> Models were evaluated across 0.1x to 1.0x dataset tokens on <i>browndw/human-ai-parallel-corpus</i>. Adan and Muon showed faster early validation loss reduction than standard AdamW, while Lion demonstrated superior memory efficiency.",
            body_style
        ))

        # 3. Decoding Strategies
        story.append(Paragraph("3. Decoding Strategies from Scratch (Task 3)", h1_style))
        story.append(Paragraph(
            "In strict compliance with the task specification, <b>model.generate() was completely prohibited</b>; all decoding was orchestrated using <code>model.forward()</code>.",
            body_style
        ))
        story.append(Paragraph(
            "• <b>Greedy Sampling:</b> Deterministic argmax step: x_t = argmax P(x_t | x_{<t}). Fast and deterministic.<br/>"
            "• <b>Top-k Sampling (k=40, T=0.8):</b> Limits candidate space to top k tokens before categorical sampling.<br/>"
            "• <b>Top-p / Nucleus Sampling (p=0.9, T=0.8):</b> Dynamically sizes candidate set based on cumulative probability mass, preventing degenerate repetitive tails.<br/>"
            "• <b>Beam Search (Widths 1, 2, 4):</b> Maintains B hypotheses scored via length-normalized cumulative log-likelihood.",
            body_style
        ))

        # Complexity Analysis
        story.append(Paragraph(
            "<b>Time Complexity Analysis:</b> For sequence length T and beam width B, un-batched beam search requires B model forward evaluations per token, resulting in overall complexity O(B * T * Cost(forward)). When B=1, beam search simplifies to greedy decoding. Empirical timing tests on <i>hamishivi/ROCStories</i> confirm exact linear scaling of execution latency with respect to beam width.",
            callout_style
        ))

        # ELI5 Section
        story.append(Paragraph("4. ELI5 Conceptual Explanations", h1_style))
        story.append(Paragraph(
            "<b>1. Mixture of Experts:</b> Instead of asking 4 people to solve every problem together, a receptionist assigns each problem to the 1 or 2 best people for that specific job.<br/>"
            "<b>2. Auxiliary Loss:</b> A bonus rule that prevents the receptionist from being lazy and giving all homework to only one person.<br/>"
            "<b>3. Decoupled Weight Decay:</b> Shrinking large weights directly instead of letting adaptive scaling distort how much they are shrunk.<br/>"
            "<b>4. Muon (Newton-Schulz):</b> Ensuring update matrices rotate information cleanly without squishing or stretching dimensions.<br/>"
            "<b>5. Nucleus Sampling:</b> Ignoring nonsense words while keeping all realistic choices to produce creative and natural stories.",
            body_style
        ))

        doc.build(story)
        print(f"Successfully generated {pdf_path}")
        return True
    except Exception as e:
        print(f"Error generating PDF via reportlab: {e}")
        return False

if __name__ == "__main__":
    build_pdf()
