"""CLI entry point for the factory — argparse subcommands wrapping library functions."""

from __future__ import annotations

from factory.cli._helpers import CEO_MODES as CEO_MODES
from factory.cli._helpers import RUN_MODES as RUN_MODES
from factory.cli._main import build_parser as build_parser
from factory.cli._main import main as main
from factory.cli._tmux_commands import (
    cmd_tmux as cmd_tmux,
)
from factory.cli._tmux_commands import (
    cmd_tmux_capture as cmd_tmux_capture,
)
from factory.cli._tmux_commands import (
    cmd_tmux_ls as cmd_tmux_ls,
)
from factory.cli._tmux_commands import (
    cmd_tmux_stop as cmd_tmux_stop,
)
from factory.cli.admin import (
    cmd_config as cmd_config,
)
from factory.cli.admin import (
    cmd_detect as cmd_detect,
)
from factory.cli.admin import (
    cmd_discover as cmd_discover,
)
from factory.cli.admin import (
    cmd_emit as cmd_emit,
)
from factory.cli.admin import (
    cmd_home as cmd_home,
)
from factory.cli.admin import (
    cmd_init as cmd_init,
)
from factory.cli.admin import (
    cmd_install as cmd_install,
)
from factory.cli.admin import (
    cmd_log as cmd_log,
)
from factory.cli.admin import (
    cmd_notify as cmd_notify,
)
from factory.cli.admin import (
    cmd_profile as cmd_profile,
)
from factory.cli.admin import (
    cmd_self_update as cmd_self_update,
)
from factory.cli.admin import (
    cmd_study as cmd_study,
)
from factory.cli.admin import (
    cmd_usage as cmd_usage,
)
from factory.cli.agents import (
    cmd_ace as cmd_ace,
)
from factory.cli.agents import (
    cmd_ace_stats as cmd_ace_stats,
)
from factory.cli.agents import (
    cmd_agent as cmd_agent,
)
from factory.cli.agents import (
    cmd_runners_list as cmd_runners_list,
)
from factory.cli.backlog import (
    cmd_backlog_add as cmd_backlog_add,
)
from factory.cli.backlog import (
    cmd_backlog_list as cmd_backlog_list,
)
from factory.cli.backlog import (
    cmd_backlog_remove as cmd_backlog_remove,
)
from factory.cli.ceo import (
    cmd_ceo as cmd_ceo,
)
from factory.cli.ceo import (
    cmd_refactory as cmd_refactory,
)
from factory.cli.contained import (
    cmd_contained as cmd_contained,
)
from factory.cli.dead_code import cmd_dead_code as cmd_dead_code
from factory.cli.eval_cmds import (
    cmd_adversarial_state as cmd_adversarial_state,
)
from factory.cli.eval_cmds import (
    cmd_baseline as cmd_baseline,
)
from factory.cli.eval_cmds import (
    cmd_eval as cmd_eval,
)
from factory.cli.eval_cmds import (
    cmd_guard as cmd_guard,
)
from factory.cli.eval_cmds import (
    cmd_precheck as cmd_precheck,
)
from factory.cli.graph import (
    cmd_graph_explain as cmd_graph_explain,
)
from factory.cli.graph import (
    cmd_graph_extract as cmd_graph_extract,
)
from factory.cli.graph import (
    cmd_graph_path as cmd_graph_path,
)
from factory.cli.graph import (
    cmd_graph_query as cmd_graph_query,
)
from factory.cli.graph import (
    cmd_graph_status as cmd_graph_status,
)
from factory.cli.graph import (
    cmd_graph_update as cmd_graph_update,
)
from factory.cli.infra import (
    cmd_archive as cmd_archive,
)
from factory.cli.infra import (
    cmd_backfill_archive as cmd_backfill_archive,
)
from factory.cli.infra import (
    cmd_checkpoint as cmd_checkpoint,
)
from factory.cli.infra import (
    cmd_dashboard as cmd_dashboard,
)
from factory.cli.infra import (
    cmd_resume as cmd_resume,
)
from factory.cli.infra import (
    cmd_serve_mcp as cmd_serve_mcp,
)
from factory.cli.infra import (
    cmd_vault_init as cmd_vault_init,
)
from factory.cli.mempalace import cmd_mempalace as cmd_mempalace
from factory.cli.registry import (
    cmd_digest as cmd_digest,
)
from factory.cli.registry import (
    cmd_insights as cmd_insights,
)
from factory.cli.registry import (
    cmd_registry_list as cmd_registry_list,
)
from factory.cli.registry import (
    cmd_report_update as cmd_report_update,
)
from factory.cli.research import (
    cmd_backfill_citations as cmd_backfill_citations,
)
from factory.cli.research import (
    cmd_leakage_check as cmd_leakage_check,
)
from factory.cli.research import (
    cmd_research as cmd_research,
)
from factory.cli.research import (
    cmd_validate_research as cmd_validate_research,
)
from factory.cli.review import (
    cmd_clean_pr as cmd_clean_pr,
)
from factory.cli.review import (
    cmd_refine_begin as cmd_refine_begin,
)
from factory.cli.review import (
    cmd_refine_complete as cmd_refine_complete,
)
from factory.cli.review import (
    cmd_refine_status as cmd_refine_status,
)
from factory.cli.review import (
    cmd_review as cmd_review,
)
from factory.cli.run import (
    cmd_run as cmd_run,
)
from factory.cli.spec import (
    cmd_spec_apply_diff as cmd_spec_apply_diff,
)
from factory.cli.spec import (
    cmd_spec_generate as cmd_spec_generate,
)
from factory.cli.spec import (
    cmd_spec_impact as cmd_spec_impact,
)
from factory.cli.spec import (
    cmd_spec_scope as cmd_spec_scope,
)
from factory.cli.spec import (
    cmd_spec_update as cmd_spec_update,
)
from factory.cli.spec import (
    cmd_spec_validate as cmd_spec_validate,
)
from factory.cli.store import (
    cmd_begin as cmd_begin,
)
from factory.cli.store import (
    cmd_diff as cmd_diff,
)
from factory.cli.store import (
    cmd_explain as cmd_explain,
)
from factory.cli.store import (
    cmd_export as cmd_export,
)
from factory.cli.store import (
    cmd_finalize as cmd_finalize,
)
from factory.cli.store import (
    cmd_history as cmd_history,
)
from factory.cli.store import (
    cmd_message as cmd_message,
)
from factory.cli.store import (
    cmd_status as cmd_status,
)
from factory.cli.store import (
    cmd_summary as cmd_summary,
)
from factory.cli.telemetry import cmd_telemetry as cmd_telemetry

if __name__ == "__main__":
    raise SystemExit(main())
