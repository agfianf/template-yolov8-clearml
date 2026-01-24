import signal


class TimeoutError(Exception):
    def __init__(self, message="Function execution timed out"):
        self.message = message
        super().__init__(self.message)

def timeout_this(seconds=10, default=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):
                raise TimeoutError("Function execution timed out")

            # Set the signal handler
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)  # Start the timer

            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                result = default
            finally:
                signal.alarm(0)  # Reset the timer

            return result

        return wrapper

    return decorator
