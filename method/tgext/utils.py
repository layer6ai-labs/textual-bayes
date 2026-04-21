from textgrad.config import SingletonBackwardEngine
from textgrad.engine import EngineLM, get_engine


def standardize_engine(engine: EngineLM | str | None = None):
    """Standardize the TextGrad engine, which can generally be set via 3 formats.

    The three ways of setting the TextGrad engine for an object are:
    - As an EngineLM object
    - As a string (to be passed into textgrad.engine.get_engine)
    - By setting the engine via textgrad.set_backward_engine and then instantiating with None

    This function's contents are sourced from textgrad.model.BlackboxLLM.__init__.
    """
    if (engine is None) and (SingletonBackwardEngine().get_engine() is None):
        raise Exception(
            "No engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine."
        )
    elif engine is None:
        engine = SingletonBackwardEngine().get_engine()
    if isinstance(engine, str):
        engine = get_engine(engine)

    return engine
