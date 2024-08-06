from dataclasses import dataclass
from typing import List, Optional, Union

import dspy
import dspy.utils
from dspy.signatures.signature import SignatureMeta
from dspy.utils.dummies import dummy_rm


def test_thought_and_action_in_one_completion():
    lm = dspy.utils.DummyLM(
        [
            """Inital thoughts\nAction 1:Finish[the awnsers are:\n\n
               1. [bob](bob.com)
               2. [alice](alice.com)]""",
        ]
    )
    dspy.settings.configure(lm=lm, rm=dummy_rm())
    program = dspy.ReAct("question -> answer")
    question = "What are some links associated with alice and bob?"
    result = program(question=question)
    assert (
        result.answer
        == """the awnsers are:\n\n
               1. [bob](bob.com)
               2. [alice](alice.com)"""
    )


def test_multiline_markdown_finish_action():
    lm = dspy.utils.DummyLM(
        [
            "Inital thoughts",
            """Finish[the awnsers are:\n\n
               1. [bob](bob.com)
               2. [alice](alice.com)]""",
        ]
    )
    dspy.settings.configure(lm=lm, rm=dummy_rm())
    program = dspy.ReAct("question -> answer")
    question = "What are some links associated with alice and bob?"
    result = program(question=question)
    assert (
        result.answer
        == """the awnsers are:\n\n
               1. [bob](bob.com)
               2. [alice](alice.com)"""
    )


def test_multiline_finish_action():
    lm = dspy.utils.DummyLM(
        [
            "Inital thoughts",
            """Finish[the awnsers are:\n
               1. bob
               2. alice]""",
        ]
    )
    dspy.settings.configure(lm=lm, rm=dummy_rm())
    program = dspy.ReAct("question -> answer")
    question = "What are some common names used in security testing?"
    result = program(question=question)
    assert (
        result.answer
        == """the awnsers are:\n
               1. bob
               2. alice"""
    )


def test_example_no_tools():
    # Createa a simple dataset which the model will use with the Retrieve tool.
    lm = dspy.utils.DummyLM(
        [
            "Initial thoughts",  # Thought_1
            "Finish[blue]",  # Action_1
        ]
    )
    dspy.settings.configure(lm=lm, rm=dummy_rm())

    program = dspy.ReAct("question -> answer")

    # Check default tools
    assert isinstance(program.tools["Finish"], dspy.Example)

    # Call the ReAct module on a particular input
    question = "What is the color of the sky?"
    result = program(question=question)
    assert result.answer == "blue"

    # For debugging
    print("---")
    for row in lm.history:
        print(row["prompt"])
        print("Response:", row["response"]["choices"][0]["text"])
        print("---")

    assert lm.get_convo(-1).endswith(
        "Question: What is the color of the sky?\n" "Thought 1: Initial thoughts\n" "Action 1: Finish[blue]"
    )


def test_example_search():
    # Createa a simple dataset which the model will use with the Retrieve tool.
    lm = dspy.utils.DummyLM(
        [
            "Initial thoughts",  # Thought_1
            "Search[the color of the sky]",  # Thought_1
            "More thoughts",  # Thought_2
            "Finish[blue]",  # Action_2
        ]
    )
    rm = dummy_rm(
        [
            "We all know the color of the sky is blue.",
            "Somethng about the sky colors",
            "This sentence is completely irellevant to answer the question.",
            "Let's add some more sentences to act as summy passages.",
            "Let's add some more sentences to act as summy passages.",
            "Let's add some more sentences to act as summy passages.",
        ]
    )
    dspy.settings.configure(lm=lm, rm=rm)

    program = dspy.ReAct("question -> answer")

    # Check default tools
    assert len(program.tools) == 2
    assert isinstance(program.tools["Search"], dspy.Retrieve)
    assert isinstance(program.tools["Finish"], dspy.Example)

    # Call the ReAct module on a particular input
    question = "What is the color of the sky?"
    result = program(question=question)
    assert result.answer == "blue"

    # For debugging
    print(lm.get_convo(-1))

    assert lm.get_convo(-1).endswith(
        "Question: What is the color of the sky?\n\n"
        "Thought 1: Initial thoughts\n\n"
        "Action 1: Search[the color of the sky]\n\n"
        "Observation 1:\n"
        "[1] «We all know the color of the sky is blue.»\n"
        "[2] «Somethng about the sky colors»\n"
        "[3] «This sentence is completely irellevant to answer the question.»\n\n"
        "Thought 2: More thoughts\n\n"
        "Action 2: Finish[blue]"
    )


class DummyTool1:
    name = "Tool1"
    input_variable = "query"
    desc = ""
    num_calls = 0

    def __call__(self, *args, **kwargs):
        # test case with no passages attribute
        assert args[0] == "foo"
        self.num_calls += 1
        return "tool 1 output"


@dataclass
class DummyOutput:
    passages: str


class DummyTool2:
    name = "Tool2"
    input_variable = "query"
    desc = ""
    num_calls = 0

    def __call__(self, *args, **kwargs):
        # test case with passages attribute
        assert args[0] == "bar"
        self.num_calls += 1
        return DummyOutput(passages="tool 2 output")


def test_custom_tools():
    lm = dspy.utils.DummyLM(
        [
            "Initial thoughts",
            "Tool1[foo]",
            "More thoughts",
            "Tool2[bar]",
            "Even more thoughts",
            "Finish[baz]",
        ]
    )
    dspy.settings.configure(lm=lm)

    tool1 = DummyTool1()
    tool2 = DummyTool2()
    program = dspy.ReAct("question -> answer", tools=[tool1, tool2])

    question = "What is the color of the sky?"
    result = program(question=question)
    assert result.answer == "baz"

    # each tool should be called only once
    assert tool1.num_calls == 1
    assert tool2.num_calls == 1
    assert lm.get_convo(-1).endswith(
        "Question: What is the color of the sky?\n\n"
        "Thought 1: Initial thoughts\n\n"
        "Action 1: Tool1[foo]\n\n"
        "Observation 1: tool 1 output\n\n"
        "Thought 2: More thoughts\n\n"
        "Action 2: Tool2[bar]\n\n"
        "Observation 2: tool 2 output\n\n"
        "Thought 3: Even more thoughts\n\n"
        "Action 3: Finish[baz]"
    )


def test_signature_instructions():
    class ExampleSignature(dspy.Signature, metaclass=SignatureMeta):
        """You are going to generate output based on input."""

        input = dspy.InputField()
        output = dspy.OutputField()

    react = dspy.ReAct(ExampleSignature)

    assert react.react[0].signature.instructions is not None
    assert react.react[0].signature.instructions.startswith("You are going to generate output based on input.")
