import setuptools

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pretty-print-tensor",
    version="0.0.1",
    author="Khaled Essam",
    author_email="ichaled@gmail.com",
    description="A package to pretty-print pytorch tensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KhaledEssam/pretty-print-tensor",
    project_urls={
        "Bug Tracker": "https://github.com/KhaledEssam/pretty-print-tensor/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["prettytable"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
