<p align="center">
  <img src="/remark_logo.jpeg" alt="remark. Logo" width="40%" height=auto>
  <br>
  Michelle Dong, Caden Cheng, Julia Jin, David Wan
</p>

## About Our Project & Significance
remark—stylized as “remark.”—is a multi-agent operating system that marks down information based on remarks made in conversation. For example, remark detects plans to grab dinner in a conversation between two friends and automatically adds the event to your calendar. remark can also keep track of grocery lists for you, take notes, perform live translation, and so much more. remark selects and utilizes the optimal AI agent to carry out your everyday needs, making your life so much more convenient by helping you anywhere and everywhere. remark works in the background, almost unnoticed, but there is nothing unremarkable about the benefits it brings. remark facilitates convenience without the risk of forgetting.

## Future Expansions & Applications
We integrated remark to work with Meta Ray Bans. In the future, we hope to assimilate remark into other technology as well so that our AI assistant can work in the background to make lives so much more convenient. You don’t have to do anything—our program will mark it down for you. Here are some expansions/applications of our project:

remark makes planning more convenient. Not only does it detect if you’ve discussed and made plans in conversations but it also accounts for travel times and overlapping event times and sends you a notification when you might be late or cannot make it to an event. Furthermore, if you don’t agree to a plan or cancel an activity, remark does that for you too.

remark bridges the language gap and facilitates communication between two people who speak different languages. remark hears another person speaking in a language incomprehensible to the user; in these cases, remark will transcribe what the speaker is saying, translating—noting intent too, not merely what the person literally is saying—and reading it back to the user. remark mutes the other person while it’s translating, so the user only hears the translated speech of the person they’re conversing with.
remark also has many other capabilities, such as (but not limited to) assisting with note-taking, keeping track of your grocery or shopping list, or getting you live baking advice (e.g. mix more because it doesn’t look “light and fluffy” yet).

## What Makes Us Unique
Especially during this era of new technology and innovations, privacy concerns remain a large anxiety. Although there is existing technology that does components of what remark’s features do, we not only combine everything into one easy-to-use multi-agent operating system, but we also circumvents the privacy concerns that that existing technology comes with. By operating as a multi-agent platform, we direct each type of usage to the optimal agent to handle it and in that way make our technology extendable and adaptable to unique situations, even niche ones not encountered often and/or before. Furthermore, remark generally does not store private information. We only store temporary data (for example we have no extended live stream or video footage; instead, we use object detection for visual information), and if we do store information over a more extended period of time (e.g. however long a user may want remark to track their grocery list for them), then we would ask for consent so the system would be permission-based. In that way, remark is built around the idea of reducing privacy concerns and anxieties about security prevalent in this digital age.

# Glasses Agent — audio pipeline

Passively listens to conversation → detects plans/to-dos → creates Google
Calendar events & Tasks. Built to run against a laptop mic now and the paired
Meta Ray-Ban mic later (the glasses expose their mic as a normal Bluetooth audio
input — no Meta SDK needed for audio).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in DEEPGRAM_API_KEY + ANTHROPIC_API_KEY
```

### Google Calendar / Tasks
1. https://console.cloud.google.com → new project.
2. Enable **Google Calendar API** and **Google Tasks API**.
3. APIs & Services → Credentials → Create OAuth client ID → **Desktop app**.
4. Download as `credentials.json` into this folder.
5. First run opens a browser to authorize; a `token.json` is cached after.

## Run

```bash
python transcribe.py            # audio only — just print transcripts
python transcribe.py --devices  # list mic devices (find the glasses' index)
python main.py                  # full pipeline: mic -> Claude -> Calendar/Tasks
```

Try saying: *"Let's grab dinner at 6pm tonight."* — a few seconds after you
stop talking, the event appears in your calendar.

## Files
| file | role |
|------|------|
| `transcribe.py` | mic capture + Deepgram live streaming (the audio core) |
| `agent.py` | Claude reads transcript, decides which tool to call |
| `calendar_tool.py` | Google Calendar + Tasks API calls |
| `main.py` | wires mic → buffer → Claude → tools |

## Using the glasses
Pair the Ray-Ban Meta glasses to your machine over Bluetooth, then
`python transcribe.py --devices` to find their input index and pass
`device=<index>` to `stream_microphone(...)` in `main.py`.

## Extending (object detection, more apps)
- New external app (Spotify, Notion, reminders…): add a tool schema in
  `agent.py` `TOOLS` + a handler in `execute_tool`. Claude picks when to use it.
- Camera/object detection: requires the gated **Meta Wearables Device Access
  Toolkit** (camera frames → your phone app). Build that as a separate module
  feeding detections to the same agent.
