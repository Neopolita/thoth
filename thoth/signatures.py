import dspy


class QueryMemorySignature(dspy.Signature):
    """Query memory signature"""

    query: str = dspy.InputField()
    result: str = dspy.OutputField()


class AgentSignature(dspy.Signature):
    """
    You are a coding agent.
    You will be given a coding task.
    Make a step by step plan to solve the task, and execute it.
    When looking for information, prioritize query_memory first over web_search.
    Add all your new learnings, insights, discoveries, and notes to memory using add_memory before calling finish.
    """

    context: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    task: str = dspy.InputField()
    result: str = dspy.OutputField()
