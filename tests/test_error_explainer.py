import pytest

from agent.error_explainer import explain_exception, explain_missing_credentials


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        ("Completa SH_CLIENT_ID y SH_CLIENT_SECRET en CREDENTIALS", "MISSING_SENTINEL_HUB_CREDENTIALS"),
        ("Completa cds_api_key en CREDENTIALS", "MISSING_CDS_CREDENTIAL"),
        ("401 Unauthorized", "AUTHENTICATION_REJECTED"),
        ("403 Forbidden", "PERMISSION_DENIED"),
        ("429 Too Many Requests", "RATE_LIMIT"),
        ("ConnectionError: timed out", "NETWORK_ERROR"),
        ("No se encontraron adquisiciones", "NO_PRODUCTS"),
        ("ModuleNotFoundError: No module named rasterio", "MISSING_DEPENDENCY"),
        ("Kernel died while waiting for execute reply", "KERNEL_DIED"),
        ("MemoryError: cannot allocate memory", "OUT_OF_MEMORY"),
        ("OSError: No space left on device", "DISK_FULL"),
        ("FileNotFoundError: nodos_originales_100m.csv", "FILE_NOT_FOUND"),
        ("EmptyDataError: No columns to parse", "INVALID_OUTPUT"),
    ],
)
def test_known_errors_are_translated(message, expected_code):
    result = explain_exception(RuntimeError(message))
    assert result.code == expected_code
    assert result.recovery_steps


def test_unknown_error_does_not_echo_possible_secret():
    result = explain_exception(RuntimeError("unknown failure sk-sensitive-value"))
    rendered = result.as_chat_message()
    assert result.code == "UNCLASSIFIED_ERROR"
    assert "sk-sensitive-value" not in rendered


def test_missing_credentials_preflight_lists_names_not_values():
    result = explain_missing_credentials(["SH_CLIENT_ID", "CDS_API_KEY"])
    rendered = result.as_chat_message()
    assert result.code == "MISSING_CREDENTIALS_PREFLIGHT"
    assert "SH_CLIENT_ID" in rendered
    assert "CDS_API_KEY" in rendered
