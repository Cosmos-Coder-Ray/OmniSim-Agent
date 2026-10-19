import httpx

from omnisim.api.smoke import response_leaks


def test_response_leak_scan_checks_body_and_headers_without_echoing_values() -> None:
    response = httpx.Response(
        200,
        headers={"X-Debug": "nebius-key-value"},
        content=b"tavily-key-value",
    )

    assert response_leaks(
        response.content,
        response.headers,
        {"NEBIUS_API_KEY": "nebius-key-value", "TAVILY_API_KEY": "tavily-key-value"},
    ) == [("NEBIUS_API_KEY", "headers"), ("TAVILY_API_KEY", "body")]
