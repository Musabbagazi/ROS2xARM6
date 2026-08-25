from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, HRFlowable
import os
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(_OUT, exist_ok=True)

doc = SimpleDocTemplate(os.path.join(_OUT, "xarm6_week8_report.pdf"), pagesize=A4,
                        topMargin=1.5*cm, bottomMargin=1.4*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        title="Week 8 Report")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=17,
                       spaceAfter=2, textColor=colors.HexColor("#1a3c5e"))
sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=9,
                     textColor=colors.HexColor("#5a6b7c"), spaceAfter=2)
h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=11,
                   spaceBefore=8, spaceAfter=2,
                   textColor=colors.HexColor("#1a3c5e"))
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=9.8,
                      leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5)

S = []
S.append(Paragraph("Week 8 Report", title))
S.append(Paragraph("16 - 20 August 2026 &nbsp;|&nbsp; UFACTORY xArm 6 work cell", sub))
S.append(HRFlowable(width="100%", thickness=1,
                    color=colors.HexColor("#1a3c5e"), spaceBefore=3,
                    spaceAfter=8))

S.append(Paragraph("This Week in Short", h))
S.append(Paragraph(
    "The week split in two. The first day went to the arm: starting a fresh "
    "sorting cell and, more importantly, finding the reason the previous one "
    "could not pick. The rest went to written work - a full report of all six "
    "projects, the guide book for the arm, and the training presentation.", body))

S.append(Paragraph("A Fresh Start on the Sorting Cell", h))
S.append(Paragraph(
    "The new cell has the camera on a stand off to the side; the arm lifts the "
    "blue cubes with a suction cup and leaves the red ones. This is a restart, "
    "not a repair. The earlier fixed-camera project has the right design and "
    "passes all its own checks, but has never completed a pick, and the reason "
    "is one measurement: the link between what the camera sees and where the arm "
    "must go is off by about 17 mm, while a cup on a cube allows roughly 6 mm.", body))

S.append(Paragraph("Finding the Real Cause", h))
S.append(Paragraph(
    "The important result was tracing that error to its source, and it was not "
    "the camera. The old calibration was done by hand: the arm was left loose "
    "and I pressed the cup onto a cube twelve times, letting the vacuum sensor "
    "record each contact. That sensor confirms the cup sealed but is blind to "
    "whether it was centred - a seal happens anywhere within about 11 mm of the "
    "cube's middle. A hand also presses to a different depth every time, which "
    "showed as nearly 19 mm of variation in the recorded heights, and the wrist "
    "angle was free and never written down. Those three account for the error. "
    "The replacement removes the hand entirely: stick a cube to the cup - "
    "crookedly is fine - and let the arm drive itself through a grid of 27 "
    "positions while the camera watches the cube it is holding. The offset never "
    "changes and cancels itself out, there is no contact to press unevenly, and "
    "the wrist angle is commanded rather than guessed.", body))

S.append(Paragraph("Tools Built", h))
S.append(Paragraph(
    "An offline check suite that runs safely with no arm and no camera "
    "connected; two measurement checks that report what the camera sees without "
    "commanding a single move; a hand-jog tool; and the calibration fit itself, "
    "which now refuses to save a result that is not good enough. The previous "
    "calibration wrote a poor answer to disk with only a warning, and every tool "
    "afterwards trusted it. A repeatability gate also runs first: measure one "
    "motionless cube fifty times and check the readings agree. The first run "
    "gave 2.31 mm of spread, which the tool flags as marginal, so that is the "
    "next thing to understand.", body))

S.append(Paragraph("Documentation", h))
S.append(Paragraph(
    "A full report covering all six projects built on this arm, written as plain "
    "description with no code: what each was for, how it works, what went wrong, "
    "how it was fixed, and where it honestly stands. The xArm 6 guide book was "
    "finished and re-issued in the KACST format. From the training report I then "
    "built the King Saud University co-op presentation - 24 slides in the "
    "university theme, covering the 4-DOF arm from phase one, the six projects "
    "of phase two, the measured results, the safety practices, and the lessons "
    "and recommendations - with short video clips of the arm working.", body))

S.append(Paragraph("Where Things Stand, and Next", h))
S.append(Paragraph(
    "The taught pick-and-place and the vision picker remain the two projects "
    "proven on hardware. In the sorting cell everything except one number works: "
    "camera, cup, pump, seal sensor, arm motion, detection and both offline "
    "suites all pass, and the wrist fault that blocked the vacuum tool is solved. "
    "Calibration is the only blocker left, designed but not yet run. Next: run it "
    "behind the repeatability gate, then verify it independently with a cube in "
    "each corner against a tape measure rather than trusting the fit's own score; "
    "then teach the table surface, run detection end to end, and attempt the "
    "first stationary pick. The two drop spots for the dynamic picker are still "
    "one short teaching session away.", body))

doc.build(S)
print("PDF written")
