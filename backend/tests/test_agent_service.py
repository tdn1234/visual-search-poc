"""Tests for the agent loop using a scripted fake LLM client (no Ollama, no DB)."""

from __future__ import annotations

import asyncio
import json

from app.services import agent_tools
from app.services.agent_service import STEP_LIMIT_REPLY, AgentService


class ScriptedClient:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def chat(self, model, messages, tools):
        self.calls.append(list(messages))
        return self._replies.pop(0) if self._replies else self._last

    @property
    def _last(self):
        return {"role": "assistant", "content": "", "tool_calls": [_call("get_product", {"sku": "x"})]}


def _call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def _tool_reply(name, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [_call(name, arguments)]}


def _service(client, max_steps=5) -> AgentService:
    service = AgentService(None, None, None, agent_tools.ImageStore(), chat_client=client, max_steps=max_steps)
    service._tools["get_product"] = lambda a: {"product": {"sku": a.get("sku")}}
    service._tools["boom"] = lambda a: 1 / 0
    return service


def run(coro):
    return asyncio.run(coro)


def test_returns_final_answer_without_tools():
    client = ScriptedClient([{"role": "assistant", "content": "<think>hm</think>Hello!"}])
    result = run(_service(client).chat("hi"))
    assert result.reply == "Hello!"
    assert result.steps == []
    assert client.calls[0][0]["role"] == "system"
    assert client.calls[0][-1] == {"role": "user", "content": "hi"}


def test_tool_result_is_fed_back_to_model():
    client = ScriptedClient([
        _tool_reply("get_product", {"sku": "shoe-red"}),
        {"role": "assistant", "content": "Found it."},
    ])
    result = run(_service(client).chat("tell me about shoe-red"))
    assert result.reply == "Found it."
    assert result.steps[0]["tool"] == "get_product"
    tool_msg = client.calls[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["name"] == "get_product"
    assert json.loads(tool_msg["content"]) == {"product": {"sku": "shoe-red"}}


def test_step_cap():
    client = ScriptedClient([_tool_reply("get_product", {"sku": "a"})] * 10)
    result = run(_service(client, max_steps=3).chat("loop"))
    assert result.reply == STEP_LIMIT_REPLY
    assert len(client.calls) == 3


def test_unknown_tool_rejected_and_loop_continues():
    client = ScriptedClient([_tool_reply("delete_everything", {}), {"role": "assistant", "content": "ok"}])
    result = run(_service(client).chat("x"))
    assert "Unknown tool" in result.steps[0]["result"]["error"]
    assert result.reply == "ok"


def test_tool_exception_becomes_error_result():
    service = _service(ScriptedClient([]))
    service._tools["get_product"] = service._tools["boom"]
    assert "error" in service.run_tool("get_product", {"sku": "a"})


def test_bad_argument_shapes():
    service = _service(ScriptedClient([]))
    assert "error" in service.run_tool("get_product", "not json")
    assert "error" in service.run_tool("get_product", [1])
    assert service.run_tool("get_product", '{"sku": "a"}') == {"product": {"sku": "a"}}


def test_shopper_id_comes_from_server_not_model():
    seen = {}
    service = _service(ScriptedClient([]))
    service._tools["get_recommendations"] = lambda a: seen.update(a) or {"strategy": "popular", "products": []}
    service.run_tool("get_recommendations", {"shopper_id": "someone-else"}, shopper_id="me")
    assert seen == {"shopper_id": "me"}
    assert "error" in service.run_tool("get_recommendations", {"shopper_id": "someone-else"})
