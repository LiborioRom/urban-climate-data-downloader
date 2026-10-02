from tools.credentials import credential_status, missing_required_credentials, set_session_credentials


def test_session_credentials_are_loaded_only_into_environment(monkeypatch):
    for name in ("EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert set(missing_required_credentials()) == {"EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET"}

    configured = set_session_credentials({
        "EE_PROJECT": "project",
        "SH_CLIENT_ID": "client",
        "SH_CLIENT_SECRET": "secret",
        "CDS_API_KEY": "token",
        "UNSUPPORTED_SECRET": "must-not-be-loaded",
    })
    assert set(configured) == {"EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY"}
    assert missing_required_credentials() == []
    assert "UNSUPPORTED_SECRET" not in configured


def test_model_settings_are_not_treated_as_credentials(monkeypatch):
    for name in ("EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "another-model")
    monkeypatch.setenv("NOMINATIM_USER_AGENT", "test-client")

    assert set(credential_status()) == {
        "EE_PROJECT", "SH_CLIENT_ID", "SH_CLIENT_SECRET", "CDS_API_KEY",
    }
    assert set_session_credentials({
        "OLLAMA_MODEL": "qwen3:4b",
        "NOMINATIM_USER_AGENT": "another-client",
    }) == []
