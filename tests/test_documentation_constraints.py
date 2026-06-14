from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skill_documents_same_time_analyzer_exclusion() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    methodology_text = (ROOT / "references" / "methodology.md").read_text(encoding="utf-8")

    assert "same-time analyzer sibling" in skill_text
    assert "Exclude these from actionable system-lag analysis by default" in skill_text
    assert "Variable Eligibility" in methodology_text
    assert "Exclude diagnostic/state/quality variables from actionable lag screening by default" in methodology_text


def test_skill_uses_fde_full_model_naming() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    methodology_text = (ROOT / "references" / "methodology.md").read_text(encoding="utf-8")

    assert "Full model (FDE name" in skill_text
    assert "full vs Y-only" in skill_text
    assert "Full model (FDE name" in methodology_text
    assert "full >> Y-only" in methodology_text
