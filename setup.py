from setuptools import setup,find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "DVC-CNN-DL-DEMO"
AUTHOR_USER_NAME = "RAVIKUMARBALIJA"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = [
    "dvc==2.7.2",
    "tqdm==4.62.3"
]


setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for DVC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="bravikumar123@hotmail.com",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    license="MIT",
    python_requires=">=3.6",
    install_requires=LIST_OF_REQUIREMENTS
)
