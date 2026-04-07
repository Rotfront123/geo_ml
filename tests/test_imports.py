def test_project_imports():
    import src

    assert hasattr(src, "__version__")
