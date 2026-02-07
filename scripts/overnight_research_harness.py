#!/usr/bin/env python3
"""Prepare an autonomous overnight research run for the MNK project.

This script does not depend on external Python packages.
It renders a high-scope Codex prompt from JSON config and prepares a
run folder with reproducible context files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DEFAULT = SCRIPT_DIR.parent
DEFAULT_CONFIG = REPO_DEFAULT / "harness/overnight_config.json"
DEFAULT_TEMPLATE = REPO_DEFAULT / "harness/RESEARCH_PROMPT_TEMPLATE.md"
DEFAULT_RUN_ROOT = Path("research_runs")


@dataclass
class RunContext:
    repo: Path
    branch: str
    run_name: str
    run_dir: Path
    config: Dict[str, Any]


def run_git(repo: Path, args: Sequence[str], capture: bool = True) -> str:
    cmd = ["git", "-C", str(repo), *args]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip() if capture else ""


def ensure_branch(repo: Path, branch: str) -> None:
    existing = run_git(repo, ["branch", "--list", branch])
    if existing:
        run_git(repo, ["checkout", branch], capture=False)
    else:
        run_git(repo, ["checkout", "-b", branch], capture=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def resolve_input_path(path_arg: str, repo: Path) -> Path:
    """Resolve user path with repo-relative fallback for convenience."""
    path = Path(path_arg)
    if path.is_absolute():
        return path

    repo_relative = (repo / path).resolve()
    if repo_relative.exists():
        return repo_relative

    return path.resolve()


def build_eval_protocol_block(cfg: Dict[str, Any]) -> str:
    protocol = cfg["evaluation_protocol"]
    lines = [
        f"- {protocol['description']}",
        f"- Openings: {protocol['openings']}",
        f"- Sides per opening: {protocol['sides_per_opening']}",
        f"- Total games: {protocol['total_games']}",
        f"- Root noise: {protocol['root_noise']}",
        f"- Fixed eval sims: {protocol['fixed_eval_sims']}",
    ]
    lines.extend(f"- {note}" for note in protocol.get("notes", []))
    return "\n".join(lines)


def build_budget_block(cfg: Dict[str, Any]) -> str:
    budget = cfg["budgets"]
    return "\n".join(
        [
            f"- Max wall time (hours): {budget['max_wall_hours']}",
            f"- Max experiments: {budget['max_experiments']}",
            f"- Max GPU-hours: {budget['max_gpu_hours']}",
            f"- Stop when target reached: {budget['stop_if_target_hit']}",
        ]
    )


def build_commit_requirements_block(cfg: Dict[str, Any]) -> str:
    req = cfg["commit_requirements"]
    lines = [f"- {section}" for section in req.get("required_sections", [])]
    lines.append(f"- Include metrics in body: {req.get('require_metrics', True)}")
    lines.append(f"- Include file/artefact paths in body: {req.get('require_paths', True)}")
    return "\n".join(lines)


def render_prompt(template: str, ctx: RunContext) -> str:
    obj = ctx.config["objective"]

    replacements = {
        "{{REPO_PATH}}": str(ctx.repo),
        "{{BRANCH_NAME}}": ctx.branch,
        "{{PRIMARY_METRIC}}": str(obj["primary_metric"]),
        "{{TARGET_PERCENT}}": str(obj["target_percent"]),
        "{{EVAL_PROTOCOL}}": build_eval_protocol_block(ctx.config),
        "{{SECONDARY_CONSTRAINTS}}": format_bullets(obj.get("secondary_constraints", [])),
        "{{REFERENCE_REPOS}}": format_bullets(ctx.config.get("reference_repos", [])),
        "{{RESEARCH_SPACE}}": format_bullets(ctx.config.get("research_space", [])),
        "{{BUDGETS}}": build_budget_block(ctx.config),
        "{{COMMIT_REQUIREMENTS}}": build_commit_requirements_block(ctx.config),
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def recommended_settings(cfg: Dict[str, Any], repo: Path) -> str:
    references = cfg.get("reference_repos", [])
    lines = [
        "Codex runner settings (recommended)",
        "- Sandbox: workspace-write (not full permissions).",
        f"- Writable path: {repo}",
        "- Network: enabled (for web search and doc lookups).",
        "- Git policy: branch-isolated work only; no force-push/reset.",
        "- Extra read-only paths:",
    ]
    lines.extend(f"  - {path}" for path in references)
    lines.extend(
        [
            "- Full permissions: No, unless you need system-level installs or unrestricted filesystem writes.",
            "- If your runner supports path ACLs, grant write only to repo and read-only to reference repos.",
        ]
    )
    return "\n".join(lines)


def write_run_context(ctx: RunContext, prompt_text: str) -> None:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = ctx.run_dir / "codex_prompt.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    settings_path = ctx.run_dir / "runner_settings.md"
    settings_path.write_text(recommended_settings(ctx.config, ctx.repo), encoding="utf-8")

    state_path = ctx.run_dir / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "repo": str(ctx.repo),
                "branch": ctx.branch,
                "run_name": ctx.run_name,
                "target_metric": ctx.config["objective"]["primary_metric"],
                "target_percent": ctx.config["objective"]["target_percent"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MNK overnight research run context")
    parser.add_argument("--repo", default=".", help="Path to MNK repo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Prompt template path")
    parser.add_argument("--branch", required=True, help="Research branch to use/create")
    parser.add_argument(
        "--run-name",
        default="overnight",
        help="Run label used in research_runs/<timestamp>_<run-name>",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create/switch branch and write run files under research_runs/",
    )
    parser.add_argument("--print-prompt", action="store_true", help="Print rendered prompt")
    parser.add_argument("--show-settings", action="store_true", help="Print recommended runner settings")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"error: {repo} is not a git repository", file=sys.stderr)
        return 1

    config_path = resolve_input_path(args.config, repo)
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 1

    template_path = resolve_input_path(args.template, repo)
    if not template_path.exists():
        print(f"error: template file not found: {template_path}", file=sys.stderr)
        return 1

    config = load_json(config_path)
    template_text = template_path.read_text(encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (repo / DEFAULT_RUN_ROOT / f"{timestamp}_{args.run_name}").resolve()

    ctx = RunContext(
        repo=repo,
        branch=args.branch,
        run_name=args.run_name,
        run_dir=run_dir,
        config=config,
    )

    prompt_text = render_prompt(template_text, ctx)

    if args.prepare:
        ensure_branch(repo, args.branch)
        write_run_context(ctx, prompt_text)
        print(f"Prepared run context in {run_dir}")

    if args.show_settings:
        print(recommended_settings(config, repo))

    if args.print_prompt:
        print(prompt_text)

    if not any([args.prepare, args.show_settings, args.print_prompt]):
        print("No action requested. Use one or more of: --prepare, --show-settings, --print-prompt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
