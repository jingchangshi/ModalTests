"""Executable half of the machine-path audit for tilefoundry_demo.

The demo must run on *any* Linux server that has Git, Python, the modal client,
network access, and a valid Modal account.  Nothing about one particular
workstation may leak into the committed runtime files: no user home directory,
no conda installation, no required Modal profile name, no absolute host working
directory.

Two kinds of absolute paths are legitimate and are asserted as explicitly
permitted below, each with the reason it is exempt:

1. **Container-internal paths.**  /opt/tilefoundry, /opt/cutlass/include,
   /artifacts, /tmp, /tmp/tilefoundry-demo and /cases only exist inside a Modal
   container (or are Volume mount points there).  They are not host path
   dependencies: the host never reads or writes them.
2. **Historical provenance records.**  cases/rmsnorm_fp8/tutorial-provenance.json
   and cases/rmsnorm_fp8/reference-provenance.json are records of past runs.
   They name the checkout that produced them and must not be rewritten to look
   as if they were produced on the current machine.

The tests are deliberately textual/AST based and run with no Modal client, no
torch, no GPU, and no network, so they gate every commit.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_ROOT.parent

#: The identifier run_demo.py uses for the sibling checkout's name, used to find
#: the candidate expression in the AST without hardcoding its spelling twice.
SIBLING_MARKER = "SIBLING_CHECKOUT_NAME"


# ---------------------------------------------------------------------------
# The audit allowlists.  Every entry needs a reason; the tests assert the reason
# is non-empty so this cannot decay into a bare list of magic strings.
# ---------------------------------------------------------------------------

#: Absolute paths that exist only inside the Modal container and are therefore
#: NOT host path dependencies.  A string beginning with one of these prefixes is
#: permitted in the runtime files.
CONTAINER_INTERNAL_PATHS = {
    "/opt/tilefoundry": (
        "the image installs the uploaded TileFoundry snapshot here "
        "(Image.add_local_dir target); it exists only inside the container"
    ),
    "/opt/cutlass/include": (
        "TILEFOUNDRY_CUTLASS_INCLUDE inside the image; CUTLASS headers are "
        "baked into the container, the host has none"
    ),
    "/artifacts": (
        "the Modal Volume mount point inside the container; the host reaches "
        "the same bytes through 'modal volume get', never through this path"
    ),
    "/tmp": (
        "container-local scratch space; never a host path in any command"
    ),
    "/tmp/tilefoundry-demo": (
        "scratch directory created by the container function itself"
    ),
    "/cases": (
        "run-scoped case-upload prefix on the artifacts Volume, inside the "
        "container only"
    ),
}

#: Absolute-looking literals that are not paths at all.  Each is a bare
#: filename/naming fragment, so it cannot bind the demo to a host layout.
NON_PATH_ABSOLUTE_LITERALS = {
    "/tilefoundry-demo-": (
        "prefix of the archive basename tilefoundry-demo-<run-id>.tar.gz, "
        "not a filesystem path; it never touches a filesystem"
    ),
}

#: Tracked files that legitimately name the historical checkout.  They are
#: past-run records: rewriting them would falsify the provenance chain.
HISTORICAL_PROVENANCE_FILES = frozenset(
    {
        "cases/rmsnorm_fp8/tutorial-provenance.json",
        "cases/rmsnorm_fp8/reference-provenance.json",
    }
)

#: The one historical record that names a checkout path.  The reference record
#: names only generation parameters, so it is not checked for that.
PROVENANCE_FILE_WITH_CHECKOUT = "cases/rmsnorm_fp8/tutorial-provenance.json"

#: Reference data is regenerated inside a Modal CPU container on every run, so
#: a fresh clone must not contain (or track) it.
REFERENCE_DATA_BINARIES = ("x.pt", "expected.pt", "model.safetensors")

#: Forbidden classes of host coupling.  Each is fatal because a fresh server
#: cannot satisfy it.  The patterns name the coupling, not one user's spelling:
#: a public clone URL that happens to contain the upstream owner is not a
#: coupling, whereas '/home/<user>' or a pinned profile name is.
FORBIDDEN_PATTERNS = {
    "user home directory": re.compile(r"/home/[A-Za-z0-9_.-]+"),
    "macOS home directory": re.compile(r"/Users/[A-Za-z0-9_.-]+"),
    "root home directory": re.compile(r"/root/[A-Za-z0-9_.-]+"),
    "Windows profile path": re.compile(r"[A-Za-z]:\\Users\\"),
    "conda installation": re.compile(
        r"(?:/miniconda3|/anaconda3|/miniforge3|/opt/conda|conda\.sh"
        r"|\bconda\s+activate\b|\bconda\s+env\s+create\b|\bconda\s+install\b)"
    ),
    "named Modal profile": re.compile(
        r"\bmodal\s+profile\s+(?:activate|switch|set|use)\s+\S+"
    ),
    "pinned MODAL_PROFILE": re.compile(r"\bMODAL_PROFILE\s*=\s*\S+"),
}

#: An absolute cd in a documented command is a working-directory assumption
#: about one particular machine.  A $HOME-relative or relative cd is fine.
ABSOLUTE_CD = re.compile(r"(?m)^[ \t]*(?:\$ )?cd[ \t]+(?P<target>/(?!/)[^\s'\"]*)")


# ---------------------------------------------------------------------------
# The scan set: the tracked runtime files of the demo.  This is what a fresh
# clone actually executes or reads at run time.
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _tracked_paths() -> list[str]:
    try:
        output = _git("ls-files", "tilefoundry_demo")
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - git missing
        pytest.skip("not a git checkout; the audit needs 'git ls-files'")
    return [line for line in output.splitlines() if line]


def _runtime_relative_paths() -> list[str]:
    """Return tracked demo-relative paths that the runtime depends on.

    Included: every *.py at the demo root (the entrypoints and the modules they
    import), tasks/*.json (the task registry), the RMSNorm case sources and
    config.json, and README.md.  Excluded: tests (not shipped), the two
    historical provenance files (see HISTORICAL_PROVENANCE_FILES), and generated
    evidence under results/ and reports/ (untracked).
    """
    paths: list[str] = []
    for tracked in _tracked_paths():
        relative = Path(tracked).relative_to("tilefoundry_demo").as_posix()
        if relative in HISTORICAL_PROVENANCE_FILES:
            continue
        path = Path(relative)
        if path.name == "README.md":
            paths.append(relative)
        elif path.parent == Path(".") and path.suffix == ".py":
            paths.append(relative)
        elif path.parent == Path("tasks") and path.suffix == ".json":
            paths.append(relative)
        elif path.parent == Path("cases/rmsnorm_fp8") and (
            path.suffix == ".py" or path.name == "config.json"
        ):
            paths.append(relative)
    return sorted(paths)


def _read(relative: str) -> str:
    return (DEMO_ROOT / relative).read_text(encoding="utf-8")


def _python_literals(text: str) -> list[tuple[int, str]]:
    """Every string constant in a Python file, with its line number."""
    tree = ast.parse(text)
    literals: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append((node.lineno, node.value))
    return literals


def _iter_scanned():
    for relative in _runtime_relative_paths():
        yield relative, _read(relative)


def _raise_message(module: str, exception: str) -> str:
    """Return the message of a raise statement in a module, verbatim.

    Used so the README quotes the code's error text instead of a stale copy.
    """
    tree = ast.parse(_read(module))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        call = node.exc
        if not isinstance(call, ast.Call):
            continue
        name = getattr(call.func, "id", getattr(call.func, "attr", ""))
        if name != exception or not call.args:
            continue
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value
        # The message may be assembled from several string pieces, possibly across
        # f-string boundaries; the literal pieces are still the code's own text.
        parts = [
            piece.value
            for piece in ast.walk(argument)
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
        ]
        if parts:
            return "".join(parts)
    raise AssertionError(f"no raise {exception}(...) with a literal message in {module}")


# ---------------------------------------------------------------------------
# The scan set itself must not be able to shrink silently.
# ---------------------------------------------------------------------------
def test_scan_set_covers_every_tracked_runtime_file() -> None:
    scanned = _runtime_relative_paths()

    # Every entrypoint and every module run_demo/run_experiments import lives at
    # the demo root.  A new root-level module is covered automatically.
    tracked = {
        Path(item).relative_to("tilefoundry_demo").as_posix()
        for item in _tracked_paths()
    }
    root_python = {
        path.relative_to(DEMO_ROOT).as_posix()
        for path in DEMO_ROOT.glob("*.py")
    } & tracked
    assert root_python <= set(scanned), (
        "a tracked root-level module is missing from the scan set: "
        f"{sorted(root_python - set(scanned))}"
    )

    # The registry and the RMSNorm case are the other inputs a run needs.
    for required in (
        "README.md",
        "tasks/square.json",
        "tasks/square_cpu.json",
        "cases/rmsnorm_fp8/config.json",
        "cases/rmsnorm_fp8/rms_norm_quant.py",
        "cases/rmsnorm_fp8/rms_norm_quant_opt.py",
        "cases/rmsnorm_fp8/twin.py",
        "cases/rmsnorm_fp8/benchmark.py",
        "cases/rmsnorm_fp8/agent_candidate.py",
        "cases/rmsnorm_fp8/make_reference.py",
    ):
        assert required in scanned, f"{required} must be audited but is not in the scan set"

    # The exemption is exactly the two historical records and nothing else.
    exempt = {path for path in HISTORICAL_PROVENANCE_FILES if path in tracked}
    assert exempt == set(HISTORICAL_PROVENANCE_FILES), (
        "a historical provenance record lost track of (was it deleted?): "
        f"{sorted(set(HISTORICAL_PROVENANCE_FILES) - exempt)}"
    )
    assert not (set(scanned) & HISTORICAL_PROVENANCE_FILES)


def test_scanned_files_are_tracked_by_git() -> None:
    tracked = set(_tracked_paths())
    for relative in _runtime_relative_paths():
        assert f"tilefoundry_demo/{relative}" in tracked, (
            f"{relative} is audited but not tracked; a fresh clone would not have it"
        )


# ---------------------------------------------------------------------------
# The refusals.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", sorted(FORBIDDEN_PATTERNS))
def test_no_host_coupling_pattern(label: str) -> None:
    pattern = FORBIDDEN_PATTERNS[label]
    offenders = []
    for relative, text in _iter_scanned():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line}: {match.group(0)!r}")
    assert not offenders, f"{label} is a host coupling and must not appear: {offenders}"


def test_no_absolute_path_literal_outside_the_allowlist() -> None:
    """Every absolute string in the runtime code is an allowlisted one.

    This is the executable form of the audit: a new /home/... default, an
    absolute volume path from another server, or a hardcoded checkout location
    fails here rather than at the next reader's machine.
    """
    offenders = []
    for relative in _runtime_relative_paths():
        if not relative.endswith(".py"):
            continue  # JSON/README are covered by the textual pattern tests
        for lineno, literal in _python_literals(_read(relative)):
            if not literal.startswith("/") or literal == "/":
                continue
            if literal in NON_PATH_ABSOLUTE_LITERALS:
                continue
            if any(
                literal == root or literal.startswith(root + "/")
                for root in CONTAINER_INTERNAL_PATHS
            ):
                continue
            offenders.append(f"{relative}:{lineno}: {literal!r}")
    assert not offenders, (
        "absolute path literal(s) not on the container-internal allowlist: " + ", ".join(offenders)
    )


def test_no_absolute_working_directory_in_documented_commands() -> None:
    offenders = []
    for relative, text in _iter_scanned():
        for match in ABSOLUTE_CD.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line}: cd {match.group('target')}")
    assert not offenders, (
        "documented commands must use $HOME-relative or relative paths so the "
        f"working-directory requirement is portable: {offenders}"
    )


# ---------------------------------------------------------------------------
# What is explicitly permitted, and why.
# ---------------------------------------------------------------------------
def test_container_internal_paths_are_documented_as_not_host_dependencies() -> None:
    # (a) The allowlist is not a bare list of magic strings: every exemption has
    #     a written reason.
    for path, reason in CONTAINER_INTERNAL_PATHS.items():
        assert path.startswith("/"), path
        assert len(reason) > 20, f"{path} needs a real reason, got {reason!r}"

    # (b) The allowlist is not vacuous: the design really does use container
    #     paths, and they really are in the audited code today.
    source = "\n".join(
        _read(relative) for relative in _runtime_relative_paths() if relative.endswith(".py")
    )
    for path in ("/opt/tilefoundry", "/artifacts"):
        assert path in source, f"{path} vanished from the runtime code; revisit the allowlist"

    # (c) The README states the rule and names each permitted path, so a reader
    #     cannot mistake a container path for a machine requirement.
    readme = _read("README.md")
    lowered = readme.lower()
    assert "container-internal" in lowered
    for path in CONTAINER_INTERNAL_PATHS:
        assert path in readme, f"README must name the container-internal path {path}"


def test_historical_provenance_files_are_kept_as_past_run_records() -> None:
    tracked = set(_tracked_paths())
    readme = _read("README.md")
    for relative in sorted(HISTORICAL_PROVENANCE_FILES):
        assert f"tilefoundry_demo/{relative}" in tracked
        assert Path(relative).name in readme, (
            f"README must explain {relative} as a past-run record"
        )
    # The one file that recorded a checkout keeps that path: it is history, not
    # a dependency, and rewriting it would falsify the record.
    historical = json.loads(_read(PROVENANCE_FILE_WITH_CHECKOUT))
    assert historical["tilefoundry"]["repo"].startswith("/home/"), (
        "the tutorial provenance record must keep the checkout path it was produced from"
    )
    # The README must say those files are historical, so the audit is not read
    # as covering them.
    assert "historical" in readme.lower()


def test_no_reference_data_binary_is_tracked() -> None:
    tracked = _git("ls-files", "tilefoundry_demo").splitlines()
    offenders = [
        item
        for item in tracked
        if Path(item).name in REFERENCE_DATA_BINARIES
        or Path(item).name.endswith((".pt", ".safetensors"))
    ]
    assert not offenders, (
        "reference data is regenerated inside a Modal CPU container on every run and "
        f"must not be committed: {offenders}"
    )

    # The ignore rules are the mechanism, so they are asserted too.
    ignore = _read("cases/rmsnorm_fp8/.gitignore")
    for name in REFERENCE_DATA_BINARIES:
        assert name in ignore, f"cases/rmsnorm_fp8/.gitignore must ignore {name}"


# ---------------------------------------------------------------------------
# The prose that must exist for those refusals to be useful.
# ---------------------------------------------------------------------------
def test_readme_states_the_cross_server_requirements() -> None:
    readme = _read("README.md")
    lowered = readme.lower()

    # Only these are needed on the host.
    for phrase in ("git", "python", "modal", "network", "modal account"):
        assert phrase in lowered, f"README must state the host requirement {phrase!r}"

    # And these must NOT be needed.
    for phrase in (
        "no local gpu",
        "no local cuda",
        "no local pytorch",
        "no local tilefoundry",
    ):
        assert phrase in lowered, f"README must state plainly that {phrase!r} is not needed"

    # The Modal profile name is arbitrary; only a working login matters.
    assert "modal profile current" in readme
    assert "profile name" in lowered
    assert "arbitrary" in lowered or "any name" in lowered

    # RMSNorm inputs are regenerated remotely, not shipped.
    assert "regenerated" in lowered, "README must say reference data is regenerated per run"
    assert "modal cpu" in lowered, "README must say it is regenerated on a Modal CPU container"

    # The checkout resolution contract: explicit override, environment variable,
    # sibling fallback, and the code's own error when none is found.
    assert "TILEFOUNDRY_REPO" in readme
    assert "sibling" in lowered
    assert "--tilefoundry-repo" in readme
    message = _raise_message("run_demo.py", "CheckoutNotFound")
    first_line = message.splitlines()[0]
    assert first_line in readme, (
        "README must quote the exact error raised when no checkout is found: "
        f"{first_line!r}"
    )
    # The recovery steps the error names must also be documented.
    assert "pyproject.toml" in readme

    # Commands are home-relative, so they are copy-pasteable on any server.
    assert "$HOME" in readme


def test_readme_documents_the_stage_status_vocabulary() -> None:
    readme = _read("README.md")
    # The vocabulary is the code's, not the README's invention: it is read from
    # the reducer's module so a rename fails this test instead of the docs.
    source = _read("demo_report.py")
    statuses = set(re.findall(r'^STATUS_[A-Z_]+ = "([A-Z_]+)"', source, re.MULTILINE))
    assert statuses, "demo_report.py should define the status constants"
    for status in sorted(statuses):
        assert status in readme, f"README must document the status {status}"


def test_readme_documents_every_stage_of_every_task() -> None:
    readme = _read("README.md")
    for path in sorted((DEMO_ROOT / "tasks").glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert spec["task_id"] in readme, f"README must document the task {spec['task_id']!r}"
        for stage in spec["stages"]:
            assert stage["stage_id"] in readme, (
                f"README must document stage_id {stage['stage_id']!r} of {path.name}"
            )
            assert stage["type"] in readme, (
                f"README must document stage type {stage['type']!r} of {path.name}"
            )


# ---------------------------------------------------------------------------
# AGENTS.md: this repository now holds two subprojects.  The triton material
# stays intact; the tilefoundry_demo section must not inherit its machine
# requirements.
# ---------------------------------------------------------------------------
AGENTS_PATH = REPO_ROOT / "AGENTS.md"
DEMO_SECTION_MARKER = "## 9 · tilefoundry_demo"


def _agents_parts() -> tuple[str, str]:
    text = AGENTS_PATH.read_text(encoding="utf-8")
    assert DEMO_SECTION_MARKER in text, (
        "AGENTS.md needs a clearly separated tilefoundry_demo section starting with "
        f"{DEMO_SECTION_MARKER!r}"
    )
    triton_part, demo_part = text.split(DEMO_SECTION_MARKER, 1)
    return triton_part, DEMO_SECTION_MARKER + demo_part


def test_agents_md_keeps_the_triton_subproject_intact() -> None:
    triton_part, _ = _agents_parts()
    for phrase in ("triton/", "study_compiler.py", "conda activate triton", "jingchangshi"):
        assert phrase in triton_part, f"the triton section lost {phrase!r}"


def test_agents_md_demo_section_is_server_independent() -> None:
    _, demo_part = _agents_parts()
    for required in (
        "tilefoundry_demo",
        "TILEFOUNDRY_REPO",
        "modal profile current",
        "no local GPU",
    ):
        assert required in demo_part, f"the tilefoundry_demo section must state {required!r}"
    for forbidden in ("conda activate triton", "jingchangshi", "miniconda", "anaconda"):
        assert forbidden not in demo_part, (
            f"{forbidden!r} must not be presented as a requirement for tilefoundry_demo"
        )
    # Prose may name the *rejected* shape ('/home/<user>'), but no concrete host
    # path may appear: a real '/home/<someone>/...' is a machine coupling.
    assert not re.search(r"/home/[A-Za-z0-9_.-]+", demo_part), (
        "the tilefoundry_demo section must not contain a concrete home path"
    )
    assert not ABSOLUTE_CD.search(demo_part), (
        "the tilefoundry_demo section must not document an absolute cd"
    )

# ---------------------------------------------------------------------------
# Drift guards: the docs describe the code's checkout resolution, not a wish.
# ---------------------------------------------------------------------------
def _sibling_checkout_name() -> str:
    """The sibling directory name run_demo.py actually looks for."""
    tree = ast.parse(_read("run_demo.py"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", None) == "SIBLING_CHECKOUT_NAME":
                return node.value.value
    raise AssertionError("run_demo.py should define SIBLING_CHECKOUT_NAME")


def _sibling_parent_steps() -> int:
    """How many 'parent' hops the sibling candidate takes from the demo directory.

    The documented layout clones both repositories under one workspace directory,
    so the checkout is beside this repository: from tilefoundry_demo/ that is two
    hops up (demo -> repo root -> workspace). Zero hops would mean "inside the
    repo"; one hop would mean the repo root itself.
    """
    tree = ast.parse(_read("run_demo.py"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "resolve_tilefoundry_repo":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if getattr(inner.func, "attr", "") != "append":
                continue
            if not inner.args or not isinstance(inner.args[0], ast.Tuple):
                continue
            pair = inner.args[0].elts
            if len(pair) != 2 or SIBLING_MARKER not in ast.dump(pair[1]):
                continue
            return sum(
                1
                for piece in ast.walk(pair[1])
                if isinstance(piece, ast.Attribute) and piece.attr == "parent"
            )
        return 0
    raise AssertionError("run_demo.py should define resolve_tilefoundry_repo()")


def _ordered_candidates() -> list[str]:
    """The candidate labels in the order the function appends them.

    ast.walk visits breadth-first, so the labels are re-sorted by source line
    to recover the execution order.
    """
    tree = ast.parse(_read("run_demo.py"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "resolve_tilefoundry_repo":
            continue
        found: list[tuple[int, str]] = []
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if getattr(inner.func, "attr", "") != "append":
                continue
            if not inner.args or not isinstance(inner.args[0], ast.Tuple):
                continue
            pair = inner.args[0].elts
            if len(pair) != 2:
                continue
            label = "".join(
                piece.value
                for piece in ast.walk(pair[0])
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
            )
            found.append((inner.lineno, label))
        return [label for _, label in sorted(found)]
    raise AssertionError("run_demo.py should define resolve_tilefoundry_repo()")



def _readme_section(heading: str) -> str:
    """Return one README section, so prose checks do not read the whole file."""
    text = _read("README.md")
    start = text.index(heading)
    following = text.find("\n## ", start + len(heading))
    return text[start:] if following == -1 else text[start:following]


def _expected_reference() -> dict:
    """The reference-data contract run_experiments.py validates against.

    Read from the source so the README cannot drift from the shapes and dtypes the
    generator is actually checked against.
    """
    tree = ast.parse(_read("run_experiments.py"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if getattr(node.targets[0], "id", None) != "EXPECTED_REFERENCE":
            continue
        return ast.literal_eval(node.value)
    raise AssertionError("run_experiments.py should define EXPECTED_REFERENCE")


def _documented_shapes(section: str) -> list[tuple[str, str]]:
    """The (shape, dtype) pairs the README writes for the reference files.

    Both bracket styles are accepted because the section writes a contract entry as
    ``[2, 7168]`` in one place and as ``(2, 56)`` in another; what matters is that
    the numbers and the dtype are the code's, not that the prose is uniform.
    """
    pattern = (
        r"[\[(]([0-9][0-9, ]*)[\])]"
        r"`?\s*(bf16|bfloat16|float8_e4m3fn|float32)"
    )
    # ``bf16`` and ``bfloat16`` are the same dtype written two ways; normalise so
    # the comparison is about the contract, not about which spelling was used.
    aliases = {"bf16": "bfloat16", "float8_e4m3fn": "float8_e4m3fn"}
    return [
        (shape.replace(" ", ""), aliases.get(dtype, dtype))
        for shape, dtype in re.findall(pattern, section)
    ]


def test_readme_documents_the_real_reference_contract() -> None:
    """A fresh clone regenerates these files, so the docs must name the real contract."""
    contract = _expected_reference()
    section = _readme_section("## 7 · The RMSNorm + FP8 experiment driver")

    # Every shape in the contract must appear in the section, written the way the
    # contract writes it. Checking the set rather than the prose keeps the test
    # independent of sentence structure.
    expected_pairs = {
        (",".join(str(dim) for dim in contract[key]["shape"]), contract[key]["dtype"].replace("torch.", ""))
        for key in ("x", "quant", "scale", "gamma")
    }
    documented = set(_documented_shapes(section))
    missing = sorted(f"{shape} {dtype}" for shape, dtype in expected_pairs - documented)
    assert not missing, (
        "README section 7 must document every shape/dtype pair in EXPECTED_REFERENCE; "
        f"missing {missing}"
    )

    # The three reference file names are what a reader has to look for, and each must
    # be named as generated rather than shipped.
    for name in ("x.pt", "expected.pt", "model.safetensors"):
        assert name in section, f"README section 7 must name {name}"
    assert "not" in section and "committed" in section, (
        "README must say the reference data is not committed"
    )


def test_readme_documents_the_real_checkout_resolution_order() -> None:
    labels = _ordered_candidates()
    assert len(labels) == 3, f"expected three candidates, the code has {labels}"
    # The three candidates are, in order: the flag, the environment variable,
    # and the sibling checkout. Assert the identity, not just the count.
    assert labels[0] == "the --tilefoundry-repo argument", labels
    assert labels[1] == "$", labels
    assert labels[2] == "the checkout beside ", labels

    # The flag, then the environment variable, then the sibling checkout: the
    # README must present them in the same order the code tries them. Read only
    # the resolution section, because the quick start mentions the variable too.
    section = _readme_section("## 4 · Which TileFoundry checkout gets used")
    markers = ["--tilefoundry-repo", "TILEFOUNDRY_REPO", "Sibling"]
    positions = []
    for marker in markers:
        assert marker in section, f"README section 4 must document the candidate {marker!r}"
        positions.append(section.index(marker))
    assert positions == sorted(positions), (
        "README section 4 must list the checkout candidates in the order the code "
        f"tries them ({markers}), got positions {positions}"
    )

    # The sibling candidate must be the directory *beside this repository*
    # (the documented layout: <workspace>/TileFoundry next to
    # <workspace>/ModalTests), which is two levels up from tilefoundry_demo/.
    # Reading the AST rather than the string keeps this honest: a fallback that
    # pointed inside the repo, or at the current directory, would fail here.
    sibling = _sibling_checkout_name()
    assert sibling == "TileFoundry"
    steps = _sibling_parent_steps()
    assert steps == 2, (
        "the sibling candidate must be derived from the directory that holds this "
        f"repository (two levels up from tilefoundry_demo/), got {steps} level(s)"
    )
    assert "beside this repository" in section or "beside this\n   repository" in section, (
        "README must state that the sibling fallback sits beside this repository"
    )
    assert "current working directory" in section, (
        "README must state that no candidate is resolved from the current working directory"
    )


def test_readme_documents_the_cpu_only_completion_scope() -> None:
    readme = _read("README.md")
    # The reason string is the code's; quote it rather than paraphrasing it.
    source = _read("demo_container.py")
    reasons = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "does not allocate a GPU" in node.value
    ]
    assert reasons, "demo_container.py should state why a GPU stage was not run"
    flat_readme = " ".join(readme.split())
    for reason in reasons:
        assert " ".join(reason.split()) in flat_readme, (
            f"README must quote the NOT_RUN reason {reason!r}"
        )
    assert "completion_scope" in readme


# ---------------------------------------------------------------------------
# Documented shell blocks must be copy-pasteable.
#
# The machine-path audit checks what a command *says*; it does not check that
# the block is valid shell. A scripted edit once double-escaped the line
# continuations in the download example, so the block collapsed into one line
# containing a literal backslash-n and anyone copying it got a broken command.
# That is a distinct failure mode from a wrong path, so it gets its own gate.
# ---------------------------------------------------------------------------
FENCE = "`" * 3
FENCED_BLOCK = re.compile(FENCE + r"([^\n]*)\n(.*?)" + FENCE, re.S)

#: Info-string tags whose fenced body is shell. A Python example, a directory tree,
#: or a quoted error message is not a command anyone pastes, so the backslash rules
#: below do not describe it and must not be applied to it: a correct Python snippet
#: containing a "\n" escape is not a broken shell continuation.
SHELL_FENCE_TAGS = frozenset({"", "bash", "sh", "shell", "console", "zsh"})


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced code block as (info-string tag, body), in document order."""
    return [(tag.strip().lower(), body) for tag, body in FENCED_BLOCK.findall(text)]


def _shell_blocks(text: str) -> list[tuple[int, str]]:
    """(document-order index, body) for each fenced block that holds shell commands.

    The index stays the position among *all* fences, not among shell fences only, so a
    failure message points at the block a reader counts in the file.
    """
    return [
        (index, body)
        for index, (tag, body) in enumerate(_fenced_blocks(text))
        if tag in SHELL_FENCE_TAGS
    ]


#: A collapsed continuation leaves *the two characters* backslash-n where a line
#: break belongs, so it is followed by the indentation of the line it swallowed.
#: A backslash-n with non-space on both sides is text the command means, such as
#: grep 'a\nb' or printf 'x\ny', which are valid shell and must not be flagged.
COLLAPSED_CONTINUATION = re.compile(r"\\n(?=[ \t])")


def _copy_paste_problems(text: str) -> list[str]:
    """Every way the shell blocks in *text* would break when pasted into a shell."""
    offenders = []
    for index, body in _shell_blocks(text):
        # A collapsed continuation leaves the two characters backslash-n where a
        # line break used to be, so it is followed by the indentation of the line
        # it swallowed. A backslash-n with non-space on both sides is text the
        # command means, such as grep 'a\nb', which is valid shell.
        for _ in COLLAPSED_CONTINUATION.findall(body):
            offenders.append(f"block {index}: collapsed line continuation")
        # A line ending in two backslashes is the double-escaped form: shell reads
        # the second backslash as an escaped backslash, not a continuation, so the
        # next line becomes a separate command.
        for line in body.splitlines():
            if line.rstrip().endswith("\\\\"):
                offenders.append(
                    f"block {index}: doubled line continuation: {line.strip()[:60]}"
                )
    return offenders


def test_documented_shell_blocks_are_copy_pasteable() -> None:
    """A documented command block must not carry escaped-newline artefacts."""
    offenders = []
    for relative in ("README.md", "../AGENTS.md"):
        path = DEMO_ROOT / relative
        if not path.exists():
            continue
        offenders += [
            f"{relative} {problem}"
            for problem in _copy_paste_problems(path.read_text(encoding="utf-8"))
        ]
    assert not offenders, (
        "documented shell blocks must be copy-pasteable, so a line continuation is a "
        f"single backslash and never a literal backslash-n: {offenders}"
    )


def test_the_copy_pasteable_gate_catches_the_escaping_bug() -> None:
    """The gate must fail on the shape it exists to catch, not just pass vacuously."""
    doubled = "```bash\nmodal volume get a \\\\n  b \\\\n  c\n```"
    collapsed = "```bash\nmodal volume get a \\n      b\n```"
    for broken in (doubled, collapsed):
        assert _copy_paste_problems(broken), f"the gate would not have caught {broken!r}"


def test_the_copy_pasteable_gate_accepts_escapes_the_command_means() -> None:
    """A real escape, with non-space on both sides, is not a collapsed continuation."""
    grep_escape = "```bash\ngrep 'a\\nb' file.txt\n```"
    printf_escape = "```bash\nprintf 'x\\ny'\n```"
    for acceptable in (grep_escape, printf_escape):
        assert not _copy_paste_problems(acceptable), (
            f"valid shell must not be flagged as a collapsed continuation: {acceptable!r}"
        )


def test_the_copy_pasteable_gate_ignores_non_shell_fences() -> None:
    """A Python or text example may legitimately hold a backslash; only shell is checked."""
    pythonish = "```python\nprint(\"a\\nb\")\n```"
    textish = "```text\nline1\\nline2\n```"
    assert _shell_blocks(pythonish) == [], "a Python example is not a shell command"
    assert _shell_blocks(textish) == [], "a text example is not a shell command"
    assert _fenced_blocks(pythonish)[0][0] == "python"
    assert not _copy_paste_problems(pythonish), "a Python example is not a shell command"
    assert not _copy_paste_problems(textish), "a text example is not a shell command"
