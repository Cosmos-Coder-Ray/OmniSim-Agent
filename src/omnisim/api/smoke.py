from collections.abc import Mapping


def response_leaks(
    body: bytes, headers: Mapping[str, str], secrets: Mapping[str, str]
) -> list[tuple[str, str]]:
    header_bytes = "\n".join(f"{name}: {value}" for name, value in headers.items()).encode()
    leaks = []
    for name, secret in secrets.items():
        value = secret.encode()
        if value in body:
            leaks.append((name, "body"))
        if value in header_bytes:
            leaks.append((name, "headers"))
    return leaks
