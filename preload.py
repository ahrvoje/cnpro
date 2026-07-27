def preload(parser):
    parser.add_argument(
        "--cnpro-loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--cnpro-tracemalloc",
        action="store_true",
        help="Enable memory tracing.",
        default=None,
    )
