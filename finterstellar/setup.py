import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="finterstellar",
    version="0.0.1",
    author="Andy KIM",
    author_email="finterstellar@naver.com",
    description="Quantitative analysis tools for investment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/finterstellar/library",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)