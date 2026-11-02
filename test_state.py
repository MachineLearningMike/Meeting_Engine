from meeting_engine.state import MeetingState
from meeting_engine.schemas import make_problem, make_proposal

problem = make_problem(
    "Trading frequency is too high"
)

proposal = make_proposal(
    problem_id=problem["id"],
    description="Investigate adaptive trading frequency"
)

print(problem)
print(proposal)