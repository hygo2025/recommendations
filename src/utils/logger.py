import logging


class Logger:
    @classmethod
    def get_logger(cls, name:str, level: int = logging.INFO):
        logger = logging.getLogger(name=name)

        logger.setLevel(level=level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger