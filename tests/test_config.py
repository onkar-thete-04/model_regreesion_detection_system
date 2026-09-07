from llm_regress.config import Config, RunnerConfig


def test_load_config_reads_runner_and_dimensions():
    config = Config.load("config.yaml")

    assert config.feature["name"] == "email_classifier"
    assert isinstance(config.runner, RunnerConfig)
    assert config.runner.max_concurrency == 8
    assert config.runner.timeout_seconds == 60
    assert config.runner.max_retries == 2
    assert config.dimensions == ["tone", "relevance", "grounding", "summary_relevance"]
