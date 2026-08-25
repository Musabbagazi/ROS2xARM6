from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                HRFlowable)

doc = SimpleDocTemplate("G:/ROS 2/xarm6_report.pdf", pagesize=A4,
                        topMargin=1.7*cm, bottomMargin=1.7*cm,
                        leftMargin=2.2*cm, rightMargin=2.2*cm,
                        title="Week 4 Report")

styles = getSampleStyleSheet()
title = ParagraphStyle("t", parent=styles["Title"], fontSize=18,
                       spaceAfter=2, textColor=colors.HexColor("#1a3c5e"))
sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=9.5,
                     textColor=colors.HexColor("#666666"), spaceAfter=6,
                     alignment=1)
h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=12,
                   spaceBefore=11, spaceAfter=3,
                   textColor=colors.HexColor("#1a3c5e"))
body = ParagraphStyle("b", parent=styles["Normal"], fontSize=10.5,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)

S = []
S.append(Paragraph("Week 4 Report", title))
S.append(HRFlowable(width="100%", thickness=1,
                    color=colors.HexColor("#1a3c5e"), spaceBefore=4,
                    spaceAfter=10))

S.append(Paragraph("Getting Started", h))
S.append(Paragraph(
    "I set out to teach a robot arm to do a job most people would find simple "
    "but a machine finds surprisingly hard: pick something up from one place "
    "and put it down in another. I began with a box. Later I got more "
    "ambitious and gave the arm a camera, so that instead of me telling it "
    "exactly where everything is, it could look at a red cube, figure out "
    "where it sat and how big it was, and go and fetch it on its own.", body))

S.append(Paragraph("Teaching It the Basics", h))
S.append(Paragraph(
    "The first version was about as simple as it gets. I showed the arm two "
    "places by moving it there myself &mdash; one where the box waits, and one "
    "where it should end up. After that, a single click was enough: the arm "
    "would reach over, close its gripper, carry the box across, let go, and "
    "settle back to its resting spot. I turned each step into a button on the "
    "desktop so I would never have to type anything to run it.", body))

S.append(Paragraph("The Rough Patches", h))
S.append(Paragraph(
    "It did not all go smoothly, and honestly that was the interesting part. "
    "Early on the arm kept freezing for no obvious reason. It took a while to "
    "realise it simply did not know how heavy its own gripper was, so it "
    "mistook that weight for a crash and stopped itself out of caution. The "
    "moment I told it what the gripper weighed, the mystery stops disappeared. "
    "Another time it locked up whenever I asked it to reach a spot right at "
    "the far edge of where it can stretch, so I taught it to move more gently "
    "and to check beforehand that a spot was actually within reach. There was "
    "even a stretch where the gripper itself sat slightly crooked, which I "
    "straightened out and set back to a clean starting angle. Every time I "
    "changed the program I checked it over carefully first, because there is a "
    "real machine on the other end and I would rather be safe than sorry.", body))

S.append(Paragraph("Giving It Eyes", h))
S.append(Paragraph(
    "The part I am most pleased with is the camera. I fitted a small one to "
    "the arm's wrist so it could hunt for the red cube by itself. Rather than "
    "measuring everything by hand, I let the arm work out the connection "
    "between what it sees and where things really are: it makes a couple of "
    "little test moves, watches how the view shifts, and teaches itself. It "
    "also sizes up each cube and adjusts how wide it opens its fingers to "
    "suit. And to make the fiddly setup less painful, I can now steer the arm "
    "around with the keyboard, a bit like a game, instead of typing in "
    "numbers.", body))

S.append(Paragraph("Where I Am Now", h))
S.append(Paragraph(
    "At this point the camera reliably finds the cube and measures it, and the "
    "arm moves safely, stopping cleanly and telling me plainly whenever "
    "something is off. There is one last step to finish before it can run "
    "completely hands-free: lining the fingers up around the cube once so the "
    "arm has a reference to work from. After that, it should be able to spot a "
    "red cube, go to it, and pick it up entirely on its own.", body))

doc.build(S)
print("PDF written: G:/ROS 2/xarm6_report.pdf")
