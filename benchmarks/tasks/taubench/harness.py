"""τ-bench retail without litellm: the benchmark's environment, tools, tasks and reward (from the repository fetch.sh
clones), with the simulated customer and the agent's model called through common.llm (cached, counted). The baseline
and the solution both run through `run_task`.

An agent is a function agent(session) that calls session.say(text) and session.call(tool, **kwargs) until
session.done. The session records every step, so the scorer sees which data-changing calls were made."""
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, chat  # noqa: E402

sys.path.insert(0, str(DATA / "taubench/repo"))
_stub = types.ModuleType("litellm")                         # the package imports litellm at load; nothing here calls it
_stub.completion = None
_stub.provider_list = []
sys.modules.setdefault("litellm", _stub)

from tau_bench.envs.retail import MockRetailDomainEnv      # noqa: E402
from tau_bench.envs.user import LLMUserSimulationEnv       # noqa: E402
from tau_bench.types import Action, RESPOND_ACTION_NAME    # noqa: E402

READS = {"find_user_id_by_name_zip", "find_user_id_by_email", "get_order_details", "get_product_details",
         "get_user_details", "list_all_product_types", "calculate", "think"}
USER_MODEL = os.environ.get("TAU_CUSTOMER", "qwen/qwen3.7-plus")   # gpt-oss-120b as the customer speaks for the agent and stops early


class Customer(LLMUserSimulationEnv):
    """The benchmark's simulated customer (its prompt, its rules), answered by an OpenRouter model."""
    def __init__(self, model=USER_MODEL):
        self.messages, self.model, self.provider, self.total_cost = [], model, None, 0.0

    def generate_next_message(self, messages):
        text = chat(self.model, messages, tag="taubench/customer", max_tokens=1500, reasoning="low")
        if not text:                                         # the model thought past its token limit: no reply is not a
            text = chat(self.model, messages, tag="taubench/customer", max_tokens=8000, reasoning="low")     # STOP
        text = text or "Sorry, could you say that again?"
        if "###STOP###" in text and text.strip() != "###STOP###":           # the benchmark's rule: STOP is a standalone
            text = text.replace("###STOP###", "").strip()                   # message; said with an answer, the answer counts
        self.messages.append({"role": "assistant", "content": text})
        return text


class Session:
    def __init__(self, index, split="test", user_model=USER_MODEL, max_steps=30):
        self.env = MockRetailDomainEnv(user_strategy="human", task_split=split, task_index=index)
        self.env.user = Customer(user_model)
        self.index, self.split, self.max_steps = index, split, max_steps
        self.steps, self.done, self.reward = [], False, 0.0
        self.wiki, self.tools = self.env.wiki, self.env.tools_info
        self.first = self.env.reset(task_index=index).observation          # the customer's first message

    def _step(self, name, kwargs):
        if self.done or len(self.steps) >= self.max_steps:
            self.done = True
            return "The conversation is over."
        r = self.env.step(Action(name=name, kwargs=kwargs))
        self.steps.append({"name": name, "kwargs": kwargs, "observation": r.observation[:2000]})
        self.done, self.reward = r.done or len(self.steps) >= self.max_steps, r.reward
        return r.observation

    def say(self, text):
        """Send a message to the customer → their reply."""
        return self._step(RESPOND_ACTION_NAME, {"content": text})

    def call(self, tool, **kwargs):
        """Call a tool → its result (a string; "Error: ..." when the environment refused)."""
        return self._step(tool, kwargs)

    def result(self):
        task = self.env.task
        if not self.env.actions or not any("###STOP###" in s["observation"] for s in self.steps) and "transfer_to_human_agents" not in [s["name"] for s in self.steps]:
            self.reward = self.env.calculate_reward().reward                 # ended by the step limit: judge the state as is
        gold = {json.dumps([a.name, a.kwargs], sort_keys=True) for a in task.actions}
        writes = [s for s in self.steps if s["name"] not in READS and s["name"] != RESPOND_ACTION_NAME]
        done_writes = [s for s in writes if not s["observation"].startswith("Error")]
        return {"id": f"retail_{self.split}_{self.index}", "reward": float(self.reward), "steps": len(self.steps),
                "writes": [[s["name"], s["kwargs"]] for s in done_writes],
                "writes_refused_by_env": len(writes) - len(done_writes),
                "writes_not_in_gold": sum(json.dumps([s["name"], s["kwargs"]], sort_keys=True) not in gold for s in done_writes),
                "gold_writes": sum(a.name not in READS for a in task.actions),
                "trajectory": self.steps}


def run_task(index, agent, split="test", **kw):
    s = Session(index, split, **kw)
    try:
        agent(s)
        err = None
    except Exception as e:                                  # noqa: BLE001 - an agent that crashes fails the task
        err = f"{type(e).__name__}: {e}"[:300]
    out = s.result()
    if err:
        out["agent_error"] = err
    return out
