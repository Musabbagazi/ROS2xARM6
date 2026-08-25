from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                HRFlowable)

doc = SimpleDocTemplate("G:/ROS 2/xarm6_week5_report.pdf", pagesize=A4,
                        topMargin=1.7*cm, bottomMargin=1.7*cm,
                        leftMargin=2.2*cm, rightMargin=2.2*cm,
                        title="Week 5 Report")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=18,
                       spaceAfter=2, textColor=colors.HexColor("#1a3c5e"))
h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=12,
                   spaceBefore=11, spaceAfter=3,
                   textColor=colors.HexColor("#1a3c5e"))
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=10.5,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)

S = []
S.append(Paragraph("Week 5 Report", title))
S.append(HRFlowable(width="100%", thickness=1,
                    color=colors.HexColor("#1a3c5e"), spaceBefore=4,
                    spaceAfter=10))

S.append(Paragraph("This Week in Short", h))
S.append(Paragraph(
    "This week had two focuses. The first was sharpening the vision side of the "
    "project so the arm sees more reliably. The second was writing a guide book "
    "for the xArm 6 that students and trainees can follow on their own.", body))

S.append(Paragraph("Sharpening the Vision", h))
S.append(Paragraph(
    "Most of the week went into tightening the YOLO detection the arm relies on. "
    "The aim was fewer missed cubes, fewer false alarms, and steadier readings "
    "of where a cube actually sits, so the arm is not reacting to noise. The "
    "detection is noticeably more dependable now, and I am continuing to feed it "
    "more of our own captured images so it keeps improving under the lighting "
    "and background we really work in.", body))

S.append(Paragraph("A Guide Book for the Trainees", h))
S.append(Paragraph(
    "Alongside that, I started putting together a guide book for the xArm 6 "
    "aimed at students and new trainees. The idea is a single place that covers "
    "the basics: how to power up and connect to the arm safely, how to move it "
    "and teach it positions, how to run the ready-made buttons we set up, and "
    "what to do when it stops or reports an error. It is written plainly, with "
    "step-by-step instructions, so someone new can get the arm working without "
    "having to be walked through it in person.", body))

S.append(Paragraph("Next Week", h))
S.append(Paragraph(
    "Carry on refining the vision with more training data, and finish and tidy "
    "up the guide book so it can be handed to the trainees.", body))

doc.build(S)
print("PDF written: G:/ROS 2/xarm6_week5_report.pdf")
