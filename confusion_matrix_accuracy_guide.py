#!/usr/bin/env python3
"""
Confusion Matrix Accuracy Calculation Guide
============================================
Generates a PDF document explaining how to calculate metrics from confusion matrix.
"""

import os
import numpy as np
from datetime import datetime
from pathlib import Path

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    print("[!] Installing reportlab...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'reportlab', '-q'])
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True


def create_confusion_matrix_guide_pdf():
    """Generate PDF explaining confusion matrix metrics"""
    
    output_path = Path("output_graphs/Confusion_Matrix_Accuracy_Guide.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, 
                           rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1976D2')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#333333')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        leading=16
    )
    
    formula_style = ParagraphStyle(
        'Formula',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=15,
        spaceBefore=10,
        alignment=TA_CENTER,
        backColor=colors.HexColor('#F5F5F5'),
        borderPadding=10
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=10,
        backColor=colors.HexColor('#2D2D2D'),
        textColor=colors.white,
        borderPadding=8
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("Confusion Matrix Accuracy Guide", title_style))
    elements.append(Paragraph("Baby Cry Classification Project", styles['Heading3']))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Section 1: What is a Confusion Matrix?
    elements.append(Paragraph("1. What is a Confusion Matrix?", heading_style))
    elements.append(Paragraph(
        "A confusion matrix is a table that visualizes the performance of a classification model. "
        "Each row represents the <b>actual class</b>, and each column represents the <b>predicted class</b>. "
        "The diagonal elements show correct predictions, while off-diagonal elements show misclassifications.",
        body_style
    ))
    
    # Example confusion matrix
    elements.append(Paragraph("<b>Example: Binary Classification</b>", body_style))
    
    cm_data = [
        ['', 'Predicted: Positive', 'Predicted: Negative'],
        ['Actual: Positive', 'TP = 50 (True Positive)', 'FN = 10 (False Negative)'],
        ['Actual: Negative', 'FP = 5 (False Positive)', 'TN = 35 (True Negative)'],
    ]
    
    cm_table = Table(cm_data, colWidths=[1.5*inch, 2*inch, 2*inch])
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#E3F2FD')),
        ('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#C8E6C9')),  # TP - green
        ('BACKGROUND', (2, 2), (2, 2), colors.HexColor('#C8E6C9')),  # TN - green
        ('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#FFCDD2')),  # FN - red
        ('BACKGROUND', (1, 2), (1, 2), colors.HexColor('#FFCDD2')),  # FP - red
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.gray),
    ]))
    elements.append(cm_table)
    elements.append(Spacer(1, 20))
    
    # Section 2: Key Metrics
    elements.append(Paragraph("2. Key Metrics from Confusion Matrix", heading_style))
    
    # Accuracy
    elements.append(Paragraph("<b>Accuracy</b> - Overall correctness of the model", body_style))
    elements.append(Paragraph(
        "<font face='Courier' size='12'>Accuracy = (TP + TN) / (TP + TN + FP + FN)</font>",
        formula_style
    ))
    elements.append(Paragraph(
        "Example: (50 + 35) / (50 + 35 + 5 + 10) = 85/100 = <b>85%</b>",
        body_style
    ))
    
    # Precision
    elements.append(Paragraph("<b>Precision</b> - When model predicts positive, how often is it correct?", body_style))
    elements.append(Paragraph(
        "<font face='Courier' size='12'>Precision = TP / (TP + FP)</font>",
        formula_style
    ))
    elements.append(Paragraph(
        "Example: 50 / (50 + 5) = 50/55 = <b>90.9%</b>",
        body_style
    ))
    
    # Recall
    elements.append(Paragraph("<b>Recall (Sensitivity)</b> - Of all actual positives, how many were found?", body_style))
    elements.append(Paragraph(
        "<font face='Courier' size='12'>Recall = TP / (TP + FN)</font>",
        formula_style
    ))
    elements.append(Paragraph(
        "Example: 50 / (50 + 10) = 50/60 = <b>83.3%</b>",
        body_style
    ))
    
    # F1 Score
    elements.append(Paragraph("<b>F1 Score</b> - Harmonic mean of Precision and Recall", body_style))
    elements.append(Paragraph(
        "<font face='Courier' size='12'>F1 = 2 × (Precision × Recall) / (Precision + Recall)</font>",
        formula_style
    ))
    elements.append(Paragraph(
        "Example: 2 × (0.909 × 0.833) / (0.909 + 0.833) = <b>86.9%</b>",
        body_style
    ))
    
    elements.append(Spacer(1, 15))
    
    # Section 3: Multi-class Extension
    elements.append(Paragraph("3. Multi-Class Classification (Baby Cry)", heading_style))
    elements.append(Paragraph(
        "For multi-class problems like baby cry classification (8 classes), we calculate metrics "
        "for each class using a One-vs-Rest approach, then average them.",
        body_style
    ))
    
    # Multi-class formulas
    avg_methods = [
        ['Averaging Method', 'Description', 'When to Use'],
        ['Macro Average', 'Simple average across classes', 'All classes equally important'],
        ['Weighted Average', 'Average weighted by class support', 'Imbalanced datasets'],
        ['Micro Average', 'Aggregate TP, FP, FN globally', 'Overall performance'],
    ]
    
    avg_table = Table(avg_methods, colWidths=[1.5*inch, 2.5*inch, 2*inch])
    avg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
    ]))
    elements.append(avg_table)
    elements.append(Spacer(1, 15))
    
    # For baby cry specifically
    elements.append(Paragraph("<b>Baby Cry Classes:</b>", body_style))
    baby_classes = ['cold_cry', 'discomfort_cry', 'distress_cry', 'hungry_cry', 
                    'normal_cry', 'pain_cry', 'sleepy_cry', 'tired_cry']
    
    class_table_data = [['#', 'Class Name', 'ID']]
    for i, cls in enumerate(baby_classes):
        class_table_data.append([str(i+1), cls, str(i)])
    
    class_table = Table(class_table_data, colWidths=[0.5*inch, 2*inch, 0.7*inch])
    class_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8F5E9')]),
    ]))
    elements.append(class_table)
    elements.append(Spacer(1, 20))
    
    # Section 4: Python Code
    elements.append(Paragraph("4. Python Implementation", heading_style))
    elements.append(Paragraph(
        "Use scikit-learn to easily compute all metrics from a confusion matrix:",
        body_style
    ))
    
    code_text = """
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# After model prediction
y_true = [...]  # Ground truth labels
y_pred = [...]  # Model predictions

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# All metrics at once
print(classification_report(y_true, y_pred, target_names=class_names))

# Individual metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)
    """
    
    # Use a simpler approach for code
    elements.append(Paragraph("<pre>" + code_text + "</pre>", styles['Code']))
    elements.append(Spacer(1, 20))
    
    # Section 5: Interpretation Tips
    elements.append(Paragraph("5. Interpretation Guidelines", heading_style))
    
    tips = [
        ("High Accuracy + Low Recall", "Model may be missing important cases (e.g., missing pain cries)"),
        ("High Precision + Low Recall", "Conservative model - few false alarms but misses some cases"),
        ("High Recall + Low Precision", "Aggressive model - catches all cases but many false alarms"),
        ("Balanced F1 Score", "Good trade-off between precision and recall"),
    ]
    
    tips_data = [['Scenario', 'Interpretation']]
    for scenario, interp in tips:
        tips_data.append([scenario, interp])
    
    tips_table = Table(tips_data, colWidths=[2.5*inch, 3.5*inch])
    tips_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF9800')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FFF3E0')]),
    ]))
    elements.append(tips_table)
    elements.append(Spacer(1, 25))
    
    # Section 6: Target Metrics for Baby Cry Model
    elements.append(Paragraph("6. Target Metrics for Baby Cry Model", heading_style))
    
    target_data = [
        ['Metric', 'Target', 'Acceptable', 'Current'],
        ['Accuracy', '≥ 90%', '≥ 85%', 'After training'],
        ['Precision', '≥ 88%', '≥ 82%', 'After training'],
        ['Recall', '≥ 88%', '≥ 82%', 'After training'],
        ['F1 Score', '≥ 88%', '≥ 82%', 'After training'],
    ]
    
    target_table = Table(target_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
    target_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#673AB7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#C8E6C9')),
        ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#FFF9C4')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.gray),
    ]))
    elements.append(target_table)
    
    # Build PDF
    doc.build(elements)
    print(f"\n[OK] PDF generated: {output_path}")
    return str(output_path)


def calculate_metrics_from_confusion_matrix(cm: np.ndarray, class_names: list = None):
    """
    Calculate all metrics from a confusion matrix.
    
    Args:
        cm: Confusion matrix (n_classes x n_classes)
        class_names: List of class names
    
    Returns:
        Dictionary with all metrics
    """
    n_classes = cm.shape[0]
    
    # Per-class metrics
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    support = np.zeros(n_classes)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        support[i] = cm[i, :].sum()
        
        # Precision
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        
        # Recall
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        
        # F1
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Weighted averages
    total_support = support.sum()
    weighted_precision = (precision * support).sum() / total_support
    weighted_recall = (recall * support).sum() / total_support
    weighted_f1 = (f1 * support).sum() / total_support
    
    results = {
        'accuracy': accuracy,
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
        },
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }
    }
    
    if class_names:
        results['class_names'] = class_names
    
    return results


def print_metrics_report(metrics: dict):
    """Print a formatted metrics report"""
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS REPORT")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    
    print("\n--- Macro Average ---")
    print(f"  Precision: {metrics['macro']['precision']:.4f}")
    print(f"  Recall:    {metrics['macro']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['macro']['f1']:.4f}")
    
    print("\n--- Weighted Average ---")
    print(f"  Precision: {metrics['weighted']['precision']:.4f}")
    print(f"  Recall:    {metrics['weighted']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['weighted']['f1']:.4f}")
    
    if 'class_names' in metrics:
        print("\n--- Per-Class Metrics ---")
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 60)
        for i, name in enumerate(metrics['class_names']):
            print(f"{name:<20} {metrics['per_class']['precision'][i]:>10.4f} "
                  f"{metrics['per_class']['recall'][i]:>10.4f} "
                  f"{metrics['per_class']['f1'][i]:>10.4f} "
                  f"{int(metrics['per_class']['support'][i]):>10}")


if __name__ == "__main__":
    # Generate the PDF guide
    pdf_path = create_confusion_matrix_guide_pdf()
    
    # Example: Calculate metrics from a sample confusion matrix
    print("\n" + "=" * 60)
    print("EXAMPLE: Baby Cry Confusion Matrix Metrics")
    print("=" * 60)
    
    # Sample confusion matrix for 8 baby cry classes
    sample_cm = np.array([
        [45,  2,  1,  1,  0,  1,  0,  0],  # cold_cry
        [ 1, 52,  2,  3,  1,  1,  0,  0],  # discomfort_cry
        [ 0,  2, 38,  1,  1,  2,  1,  0],  # distress_cry
        [ 1,  2,  1, 68,  2,  1,  2,  3],  # hungry_cry
        [ 0,  1,  1,  2, 42,  1,  1,  2],  # normal_cry
        [ 1,  1,  2,  1,  1, 35,  1,  1],  # pain_cry
        [ 0,  1,  1,  2,  1,  1, 48,  1],  # sleepy_cry
        [ 0,  0,  1,  2,  2,  1,  2, 41],  # tired_cry
    ])
    
    class_names = ['cold_cry', 'discomfort_cry', 'distress_cry', 'hungry_cry',
                   'normal_cry', 'pain_cry', 'sleepy_cry', 'tired_cry']
    
    metrics = calculate_metrics_from_confusion_matrix(sample_cm, class_names)
    print_metrics_report(metrics)
