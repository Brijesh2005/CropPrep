"""Enable ``python -m services.dataset_manager <command>`` invocation."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
