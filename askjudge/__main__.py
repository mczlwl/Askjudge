"""Allow ``python -m askjudge`` to invoke the public CLI."""

from .cli import main


raise SystemExit(main())

