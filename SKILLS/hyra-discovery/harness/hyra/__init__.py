"""Hyra-Local: a self-contained Hunyuan Research Agent (Harness) scaffold.

Design mirrors Tencent Hyra-1.0:
  ContextAgent maintains an ExperienceBank, produces inspirations -> queue.
  Multiple ProposalAgents consume inspirations, write a `solution` (with a
  runnable entry), execute it in an isolated Sandbox, and score it.
  A two-layer loop: inner optimises the solution against an Evaluator,
  outer co-evolves the Evaluator (anti reward-hacking).

Inference compute = WorkBuddy's built-in HY3 (the agent itself); the
LLMBridge abstraction allows injecting that intelligence via AgentBridge
(file-based request/response) or falling back to MockBridge for headless runs.
"""

from .experience_bank import ExperienceBank
from .sandbox import Sandbox
from .llm_bridge import LLMBridge, MockBridge, AgentBridge
from .auto_proposer import GuidedProposer
from .evaluator import Evaluator
from .agents import ContextAgent, ProposalAgent
from .harness import Harness

__version__ = "0.1.0"
