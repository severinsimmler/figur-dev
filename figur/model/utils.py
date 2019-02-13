import urllib.request


def download_model(url: str):
    # Create folder in home directory:
    home = Path.home()
    resources = Path(home, ".figur", "models")
    if not resources.exists():
        resources.mkdir()
    filepath = Path(resources, "best-model.pt")
    if not filepath.exists():    
        # Download model:
        urllib.request.urlretrieve(url, filepath)
    return filepath
