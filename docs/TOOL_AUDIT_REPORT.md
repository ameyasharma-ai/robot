# 🛠️ Robot Core — 61 Registered Tools Comprehensive Audit Report
**Architecture Review & Verification Gate (Phase 0 Baseline)**

This document provides a comprehensive mapping of every registered capability across the Robot Core platform, detailing name, family, signature parameters, permission level, timeout caps, risk classification, and error policies.

## 1. Summary Statistics
- **Total Registered Tools**: 61
- **Tool Families**: 24
- **Permission Tiers**:
  - `DEVELOPMENT`: 2 tools
  - `HARMLESS_LOCAL`: 26 tools
  - `READ_ONLY`: 32 tools
  - `SYSTEM`: 1 tools

### Family Breakdown
| Family | Tool Count | Primary Responsibilities |
| :--- | :---: | :--- |
| `scheduler` | 10 | Timers, alarms, reminders, daily briefing |
| `system` | 9 | Time, hardware diagnostics, volume, brightness, status, life expectancy |
| `perception` | 5 | Camera snapshot, scene inspection, tabletop inspection, world state |
| `music` | 5 | Playback, volume, equalization, history, status |
| `os` | 5 | App launching, desktop control, clipboard, notifications |
| `smart_home` | 3 | Home Assistant IoT devices, states, controls |
| `memory` | 3 | Persistent facts, recall, all facts, user profile |
| `motion` | 2 | 3D Avatar and robotic kinematic actions |
| `image` | 2 | Local & remote image generation, cancellation |
| `network` | 2 | Wi-Fi scanning, network speedtest |
| `web` | 2 | Fast search, page content scraping |
| `calculator` | 1 | Safe mathematical expression evaluation |
| `code` | 1 | Sandboxed script execution |
| `finance` | 1 | Real-time stock and crypto quotes |
| `git` | 1 | Workspace commit and version control |
| `history` | 1 | SQLite conversation archive search |
| `knowledge` | 1 | Wikipedia summarization and deep queries |
| `news` | 1 | Live RSS news headlines |
| `notes` | 1 | Obsidian / Markdown vault search |
| `personality` | 1 | Runtime hot-swapping of AI personalities |
| `terminal` | 1 | Sandboxed command execution |
| `translation` | 1 | Multi-lingual translation |
| `weather` | 1 | Open-Meteo meteorological forecasts |
| `wikipedia` | 1 | Direct article lookup |

## 2. Complete Tool Inventory Matrix
| Tool Name | Family | Permission | Risk | Timeout | Required Params | Properties / Schema | Error & Recovery Policy |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `alarm.cancel` | `scheduler` | `HARMLESS_LOCAL` | LOW | 10.0s | `identifier` | `identifier` | Input validation -> Safe SQLite transaction rollback |
| `alarm.list` | `scheduler` | `READ_ONLY` | LOW | 10.0s | *None* | `status` | Input validation -> Safe SQLite transaction rollback |
| `alarm.set` | `scheduler` | `HARMLESS_LOCAL` | LOW | 10.0s | `target_time` | `target_time`, `label`, `recurring` | Input validation -> Safe SQLite transaction rollback |
| `calculator.evaluate` | `calculator` | `READ_ONLY` | LOW | 10.0s | `expression` | `expression` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `camera.capture_snapshot` | `perception` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | `format`, `quality` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `character.execute_action` | `motion` | `HARMLESS_LOCAL` | LOW | 10.0s | `action` | `action`, `side`, `duration`, `intensity`, `repetitions`, `direction`, `angle`, `distance`, `emotion` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `character.stop` | `motion` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `code.run_script` | `code` | `DEVELOPMENT` | LOW | 20.0s | `code` | `code`, `language`, `timeout_seconds` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `finance.quote` | `finance` | `READ_ONLY` | LOW | 3.0s | `symbol` | `symbol` | Network timeout (3.0s cap) -> Graceful degradation message |
| `git.workspace_commit` | `git` | `DEVELOPMENT` | LOW | 10.0s | *None* | `action`, `commit_message`, `files` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `history.search` | `history` | `READ_ONLY` | LOW | 10.0s | `query` | `query`, `limit` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `homeassistant.control` | `smart_home` | `HARMLESS_LOCAL` | LOW | 10.0s | `entity_id` | `entity_id`, `action`, `value` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `homeassistant.list_devices` | `smart_home` | `READ_ONLY` | LOW | 10.0s | *None* | `domain` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `homeassistant.state` | `smart_home` | `READ_ONLY` | LOW | 10.0s | `entity_id` | `entity_id` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `image.cancel` | `image` | `HARMLESS_LOCAL` | LOW | 10.0s | `job_id` | `job_id` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `image.generate` | `image` | `HARMLESS_LOCAL` | LOW | 10.0s | `prompt` | `prompt`, `width`, `height`, `steps` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `knowledge.wikipedia` | `knowledge` | `READ_ONLY` | LOW | 6.0s | `query` | `query` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `memory.all` | `memory` | `READ_ONLY` | LOW | 10.0s | *None* | `limit` | Semantic threshold gating -> Safe vector store error fallback |
| `memory.search` | `memory` | `READ_ONLY` | LOW | 10.0s | `query` | `query`, `limit` | Semantic threshold gating -> Safe vector store error fallback |
| `memory.write` | `memory` | `HARMLESS_LOCAL` | LOW | 10.0s | `fact` | `fact` | Semantic threshold gating -> Safe vector store error fallback |
| `music.control` | `music` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | `action`, `volume`, `preset`, `query` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `music.equalizer` | `music` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | `preset` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `music.history` | `music` | `READ_ONLY` | LOW | 10.0s | *None* | `limit` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `music.play` | `music` | `HARMLESS_LOCAL` | LOW | 10.0s | `query` | `query`, `source` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `music.status` | `music` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `network.speedtest` | `network` | `READ_ONLY` | LOW | 3.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `network.wifi_scan` | `network` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `news.headlines` | `news` | `READ_ONLY` | LOW | 3.0s | *None* | `category`, `limit` | Network timeout (3.0s cap) -> Graceful degradation message |
| `notes.vault_search` | `notes` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | `action`, `title`, `content`, `tags`, `query` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `os.app_launch` | `os` | `HARMLESS_LOCAL` | LOW | 10.0s | `app_name` | `app_name`, `args` | Parameter alias fallback -> Non-zero exit code capture |
| `os.capture_screen` | `system` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | `filename` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `os.clipboard_read` | `os` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Parameter alias fallback -> Non-zero exit code capture |
| `os.clipboard_write` | `os` | `HARMLESS_LOCAL` | LOW | 10.0s | `text` | `text` | Parameter alias fallback -> Non-zero exit code capture |
| `os.display_brightness` | `system` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | `action`, `level_percent` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `os.notify_desktop` | `os` | `HARMLESS_LOCAL` | LOW | 10.0s | `message` | `title`, `message`, `urgency`, `icon` | Parameter alias fallback -> Non-zero exit code capture |
| `os.system_control` | `os` | `HARMLESS_LOCAL` | LOW | 10.0s | `action` | `action`, `value` | Parameter alias fallback -> Non-zero exit code capture |
| `perception.get_world_state` | `perception` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `perception.inspect_tabletop` | `perception` | `READ_ONLY` | LOW | 10.0s | *None* | `query` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `perception.inspect_view` | `perception` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `perception.what_do_you_see` | `perception` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `personality.switch` | `personality` | `HARMLESS_LOCAL` | LOW | 10.0s | `name` | `name` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `reminder.cancel` | `scheduler` | `HARMLESS_LOCAL` | LOW | 10.0s | `identifier` | `identifier` | Input validation -> Safe SQLite transaction rollback |
| `reminder.list` | `scheduler` | `READ_ONLY` | LOW | 10.0s | *None* | `status` | Input validation -> Safe SQLite transaction rollback |
| `reminder.schedule` | `scheduler` | `HARMLESS_LOCAL` | LOW | 10.0s | `time_expression`, `message` | `time_expression`, `message` | Input validation -> Safe SQLite transaction rollback |
| `system.calculate` | `system` | `READ_ONLY` | LOW | 10.0s | `expression` | `expression` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `system.daily_briefing` | `scheduler` | `READ_ONLY` | LOW | 10.0s | *None* | `include_news`, `include_weather` | Input validation -> Safe SQLite transaction rollback |
| `system.death_clock` | `system` | `READ_ONLY` | LOW | 7.0s | *None* | `age`, `gender`, `country`, `smoking`, `exercise_days_per_week`, `diet_quality`, `sleep_hours`, `stress_level`, `alcohol`, `bmi` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `system.hardware` | `system` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `system.list_tools` | `system` | `READ_ONLY` | LOW | 10.0s | *None* | `category` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `system.status` | `system` | `READ_ONLY` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `system.stop_listening` | `system` | `HARMLESS_LOCAL` | LOW | 10.0s | *None* | *None* | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `system.time` | `system` | `READ_ONLY` | LOW | 10.0s | *None* | `timezone_str` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `terminal.execute` | `terminal` | `SYSTEM` | LOW | 35.0s | `command` | `command`, `cwd`, `timeout_seconds` | Parameter alias fallback -> Non-zero exit code capture |
| `timer.cancel` | `scheduler` | `HARMLESS_LOCAL` | LOW | 10.0s | `identifier` | `identifier` | Input validation -> Safe SQLite transaction rollback |
| `timer.list` | `scheduler` | `READ_ONLY` | LOW | 10.0s | *None* | `status` | Input validation -> Safe SQLite transaction rollback |
| `timer.set` | `scheduler` | `HARMLESS_LOCAL` | LOW | 10.0s | `duration_seconds` | `duration_seconds`, `label`, `on_complete_tool`, `on_complete_args` | Input validation -> Safe SQLite transaction rollback |
| `translation.translate` | `translation` | `READ_ONLY` | LOW | 3.0s | `text` | `text`, `target_lang`, `source_lang` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
| `weather.forecast` | `weather` | `READ_ONLY` | LOW | 6.0s | *None* | `location` | Network timeout (3.0s cap) -> Graceful degradation message |
| `web.scrape_page` | `web` | `READ_ONLY` | LOW | 4.0s | `url` | `url`, `max_chars` | Network timeout (3.0s cap) -> Graceful degradation message |
| `web.search` | `web` | `HARMLESS_LOCAL` | LOW | 7.0s | `query` | `query`, `max_results`, `deep` | Network timeout (3.0s cap) -> Graceful degradation message |
| `wikipedia.lookup` | `wikipedia` | `READ_ONLY` | LOW | 3.0s | `query` | `query`, `lang` | Input validation via `@harden_tool_inputs` -> User friendly explanation |
