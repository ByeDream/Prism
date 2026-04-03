"""Entry point for running Prism via ``python -m prism.server``."""

import uvicorn

from prism.config import settings


def main() -> None:
    uvicorn.run(
        "prism.app:app",
        host=settings.prism_host,
        port=settings.prism_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
