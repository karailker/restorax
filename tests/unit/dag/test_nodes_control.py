from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from restorax.dag.nodes.control import ChoiceNode, ChoiceRule, PassNode


def test_pass_node_echoes_input():
    node = PassNode(id="p1", name="Pass")
    result = asyncio.run(node.execute(MagicMock(), {"data": "hello"}))
    assert result.outputs["data"] == "hello"


def test_choice_node_matches_rule():
    rule = ChoiceRule(field="width", operator="gt", value=1920, branch_index=1)
    node = ChoiceNode(id="c1", name="Choice", rules=[rule], default_branch=0)
    result = asyncio.run(node.execute(MagicMock(), {"meta": {"width": 3840}}))
    assert result.outputs["branch_index"] == 1


def test_choice_node_default_when_no_match():
    node = ChoiceNode(id="c1", name="Choice", rules=[], default_branch=2)
    result = asyncio.run(node.execute(MagicMock(), {"meta": {}}))
    assert result.outputs["branch_index"] == 2
