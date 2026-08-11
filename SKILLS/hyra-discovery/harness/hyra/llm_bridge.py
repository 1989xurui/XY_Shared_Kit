"""LLMBridge: abstraction over the inference engine.

* MockBridge  - headless placeholder, returns a stub (used to run the harness
                without live inference).
* AgentBridge - file-based request/response: the harness drops a request JSON,
                the WorkBuddy/HY3 agent reads it, writes the response back, and
                the bridge returns it. This is how the built-in HY3 (the agent
                itself) is injected as the research brain.
"""
import json
import os
import time


class LLMBridge:
    def generate(self, prompt: str, system: str = None) -> str:
        raise NotImplementedError


class MockBridge(LLMBridge):
    def __init__(self):
        self.n = 0

    def generate(self, prompt, system=None):
        self.n += 1
        return "# mock proposal (no live inference)"


class AgentBridge(LLMBridge):
    """File-based request/response bridge to the built-in HY3 brain.

    Two operating modes:

    * Attended (responder=None): drops a request JSON and *blocks* until an
      external agent (a live HY3/WorkBuddy session or a human) writes the
      matching response JSON. Used when a human is in the loop.

    * Unattended (responder=<callable>): a distilled-HY3 policy answers the
      request immediately (no external API, no blocking). The req/resp JSONs
      are still written to disk so every autonomous decision is auditable.
      This is what powers the fully unattended auto-loop.
    """

    def __init__(self, pending_dir, responder=None, poll=2, timeout=600):
        self.pending_dir = pending_dir
        os.makedirs(pending_dir, exist_ok=True)
        self.req_id = 0
        self.responder = responder      # callable(prompt, system) -> str | None
        self.poll = poll
        self.timeout = timeout

    def generate(self, prompt, system=None):
        self.req_id += 1
        req_path = os.path.join(self.pending_dir, f"req_{self.req_id}.json")
        resp_path = os.path.join(self.pending_dir, f"resp_{self.req_id}.json")
        with open(req_path, "w") as f:
            json.dump({"id": self.req_id, "system": system, "prompt": prompt}, f, indent=2)

        # Unattended: distilled-HY3 policy fulfils the request right away.
        if self.responder is not None:
            resp = self.responder(prompt, system)
            with open(resp_path, "w") as f:
                json.dump({"id": self.req_id, "response": resp}, f, indent=2)
            return resp

        # Attended: block until an external agent (HY3) fulfils the request.
        waited = 0
        while not os.path.exists(resp_path):
            time.sleep(self.poll)
            waited += self.poll
            if waited > self.timeout:
                return "# agent bridge timeout"
        with open(resp_path) as f:
            data = json.load(f)
        return data.get("response", "")
