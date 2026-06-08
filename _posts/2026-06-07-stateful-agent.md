---
layout: post
title: "stateful-agent: an agent that actually remembers"
date: 2026-06-07 12:00:00
description: "A from-scratch agent harness built around cross-session memory — a tool-use control loop over a real streaming backend (Kafka, Flink, Redis, Cassandra) with semantic recall and rolling-summary compaction, served as an API."
tags: agents llm memory systems infrastructure
categories: deep-learning
featured: false
toc:
  sidebar: left
---

Most "agent" demos are a single while-loop around an LLM with a list of tools. The interesting part — the part production agents actually struggle with — is everything around the loop: state, memory across sessions, idempotency, guardrails, serving. So I built one from the harness outward: an event-driven personal assistant on a real streaming backend, wrapped in an agent harness, and served as an API. The point was twofold — ship a demoable assistant, and learn the systems/infra layer end to end.

## Three concentric layers, one artifact

```
SERVING / DEPLOY   FastAPI -> Docker -> (k8s, roadmap)
  HARNESS          loop · tools · state · memory · compaction · signals
    APP            assistant — 5 safe tools,
                   Kafka · Flink · Redis · Cassandra memory + proactive reminders
```

- **Kafka** — every action becomes an event on a log.
- **Flink** — derives signals from the event streams.
- **Redis** — hot state: recent turns, cooldowns, rolling summary.
- **Cassandra** — durable memory timeline.
- **ChromaDB** — embedded vector index for semantic recall over the timeline.
- **LLM (Claude)** — the interface, plus summarization and decision-making.

## The loop

The harness core is small on purpose. `step()` calls the model; if it asked for tools, run them and feed the results back; otherwise return the text. A hop bound keeps a misbehaving model from looping forever, and tools are wrapped so a tool exception can never crash the loop.

```python
def step(self, user_message: str) -> str:
    history = self.hot.recent_turns(config.RECENT_TURNS)
    recalled = self.memory.search_timeline(user_message, n=3)

    system = SYSTEM_PROMPT + f"\n\nCurrent time (UTC): {dt.datetime.utcnow().isoformat()}"
    if (summary := self.hot.get_summary()):
        system += f"\n\nConversation summary so far:\n{summary}"
    if recalled:
        system += "\n\nPossibly relevant older memory:\n" + "\n".join(f"- {r['text']}" for r in recalled)

    messages = history + [{"role": "user", "content": user_message}]
    reply = ""
    for _ in range(MAX_TOOL_HOPS):
        resp = llm.complete(messages, system=system, tools=self.tool_schemas)
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = [{"type": "tool_result", "tool_use_id": b.id,
                        "content": self._run_tool(b.name, b.input)}
                       for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "user", "content": results})
            continue
        reply = "".join(b.text for b in resp.content if b.type == "text")
        break

    self.hot.add_turn("user", user_message)
    self.hot.add_turn("assistant", reply)
    self.memory.add_event("conversation", f"User: {user_message} | Assistant: {reply}")
    self._maybe_compact()
    return reply
```

The tools are deliberately small and safe: `create_reminder`, `save_note`, `list_tasks`, `summarize_today`, `draft_message`. Explicitly *not* in v1: arbitrary shell execution, browser control — the two places small agent projects rot.

## Memory: three tiers, because they answer different questions

The name is the thesis. A useful assistant has to remember across sessions, and "remember" isn't one thing:

- **Hot state (Redis)** — the recent turns and a rolling summary. Answers "what were we just talking about?" Cheap, fast, bounded.
- **Durable timeline (Cassandra)** — every exchange and tool action, append-only. Answers "what happened, ever?"
- **Semantic recall (ChromaDB)** — every event is embedded; before answering, the agent pulls the top-3 timeline entries by *meaning*. Answers "what's relevant to *this*, even if it shares no keywords?" Verified by meaning-match: the query "when do I see the doctor?" recalls a "dentist appointment Friday 3pm" note with no shared word.

### Compaction so history doesn't grow forever

Re-sending an ever-growing transcript is how agents get slow and expensive. When raw turns pass a threshold, the oldest chunk is folded into a rolling summary (the LLM's summarization role) and dropped from the hot turn list:

```python
def _maybe_compact(self) -> None:
    if self.hot.turns_count() <= config.COMPACT_THRESHOLD:
        return
    old = self.hot.pop_oldest(config.COMPACT_CHUNK)
    prev = self.hot.get_summary()
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in old)
    new_summary = llm.chat([{"role": "user", "content":
        "Update the running summary. Keep it under 150 words; retain durable facts, "
        f"preferences, decisions; drop chit-chat.\n\nExisting:\n{prev or '(none)'}"
        f"\n\nNew turns:\n{convo}"}], model=config.MODEL_FAST)
    self.hot.set_summary(new_summary)
```

Tested: 22 turns → 12 raw turns + a summary that retained the facts.

## The streaming backend, and why

Every action is published to Kafka as an event. Derived signals (e.g. tool-usage counts, due reminders) come off that log, and proactive reminders fire from it — the assistant isn't purely reactive. Everything sits behind interfaces (`MemoryStore`, `HotState`, `EventBus`) so the Phase-1 build ran on local file/JSON stores and the same code later pointed at the real backends with no logic changes.

The whole stack was stood up and tested on a rented CPU VM: Redis, Kafka (KRaft mode, no ZooKeeper), Cassandra 4.1, and Flink 1.18. The end-to-end integration test goes green: `save_note` → Cassandra, `create_reminder`, `list_tasks`, semantic recall of "dark roast coffee", 12 turns in Redis, 11 events in Cassandra, Kafka tool-usage stream, and a reminder firing from the signal layer.

## Serving

A FastAPI layer exposes `/health`, `/chat`, `/tasks`, `/metrics`, tested over HTTP (a `/chat` call saved a note and fired a reminder; `/metrics` reads tool counts off the Kafka stream). Docker artifacts (Dockerfile + a full `docker-compose.yml` for the stack) are written, and the app self-bootstraps its Cassandra schema on connect so compose comes up clean.

## Roadmap / what's not done

Being honest about the edges, because they're real:

- **Flink is underused.** The cluster runs jobs, but the signal logic currently runs as a Python consumer off Kafka, not a deployed Flink job. Porting it (actions topic → derived-signals job) needs the Flink–Kafka connector.
- **No real container run yet.** Docker *artifacts* exist and the compose file is complete, but I haven't done an actual `docker build && docker compose up` on a Docker-capable host (the dev VM was itself a container, no daemon). k8s (a kind/minikube Deployment + Service) is the next step after that.

None of that blocks the thesis — an agent with real, tiered, cross-session memory, served as an API — which works end to end today.

Code: [github.com/debtirthasaha/stateful-agent](https://github.com/debtirthasaha/stateful-agent).
