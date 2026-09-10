"""Probe queries for the empirical threshold calibration in Task 4.

The brief's floor is 3 in-scope and 2 out-of-scope probes. Twelve and five
oversample it deliberately: the gap between the two clusters is the entire
justification for the chosen threshold, and three points do not make a cluster.

The in-scope probes are one per required topic and are deliberately worded
differently from the twelve evaluation queries, so that calibration and
scoring are not measuring the same twelve strings twice.
"""

IN_SCOPE_PROBES: list[str] = [
    "Who is eligible to apply for a business loan?",
    "Show me the formula used to work out a monthly instalment.",
    "What annual fee does the credit card carry?",
    "Which identity proof is accepted when opening an account?",
    "Someone used my card without permission, what happens next?",
    "How long does it take to shut a bank account?",
    "What rate of interest applies to an education loan?",
    "Will I be charged for repaying a fixed-rate loan ahead of schedule?",
    "What happens if my average balance falls below the requirement?",
    "How much does a missed payment hurt my credit rating?",
    "Can two people hold one account together?",
    "Which account types can a person living abroad hold?",
]

OUT_OF_SCOPE_PROBES: list[str] = [
    "What is the best recipe for a chocolate sponge cake?",
    "Which team won the football World Cup in 2018?",
    "How do I replace the timing belt on a diesel engine?",
    "What is the boiling point of liquid nitrogen at sea level?",
    "Recommend a three-day hiking route in the Western Ghats.",
]
