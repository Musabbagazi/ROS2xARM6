from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, HRFlowable
import os
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(_OUT, exist_ok=True)

doc = SimpleDocTemplate(os.path.join(_OUT, "xarm6_week8_report_brief.pdf"), pagesize=A4,
                        topMargin=1.7*cm, bottomMargin=1.7*cm,
                        leftMargin=2.2*cm, rightMargin=2.2*cm,
                        title="Week 8 Report")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=17,
                       spaceAfter=2, textColor=colors.HexColor("#1a3c5e"))
sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=9,
                     textColor=colors.HexColor("#5a6b7c"), spaceAfter=2)
h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=11,
                   spaceBefore=9, spaceAfter=2,
                   textColor=colors.HexColor("#1a3c5e"))
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=10.2,
                      leading=14, alignment=TA_JUSTIFY, spaceAfter=5)

S = []
S.append(Paragraph("Week 8 Report", title))
S.append(Paragraph("16 - 20 August 2026 &nbsp;|&nbsp; UFACTORY xArm 6 work cell", sub))
S.append(HRFlowable(width="100%", thickness=1,
                    color=colors.HexColor("#1a3c5e"), spaceBefore=3,
                    spaceAfter=8))

S.append(Paragraph("The Arm", h))
S.append(Paragraph(
    "I started a fresh sorting cell - camera on a side stand, suction cup, blue "
    "cubes picked and red ones left - and found why the previous version could "
    "never pick. The link between what the camera sees and where the arm must go "
    "was off by about 17 mm, against roughly 6 mm of tolerance. The cause was the "
    "calibration being done by hand: the vacuum sensor confirms the cup sealed "
    "but cannot tell whether it was centred. The replacement lets the arm hold "
    "the cube itself and drive through a grid of positions, which removes the "
    "hand from the measurement entirely. It is built and tested offline, but not "
    "yet run on the arm.", body))

S.append(Paragraph("Documentation", h))
S.append(Paragraph(
    "I wrote a full report of all six projects built on this arm, finished the "
    "xArm 6 guide book in the KACST format, and produced the King Saud "
    "University co-op presentation from the training report.", body))

S.append(Paragraph("Next", h))
S.append(Paragraph(
    "Run the new calibration on the arm and check it against a tape measure, then "
    "teach the table surface and attempt the first pick.", body))

doc.build(S)
print("PDF written")
