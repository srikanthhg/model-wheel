from model_wheel_demo import greeting


def test_greeting() -> None:
    assert greeting("Srikanth") == "Hello Srikanth, the model-wheel package is working."
