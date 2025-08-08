from setuptools import setup, find_packages

# Read the contents of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="parse-any-doc",  
    version="0.1.0",
    author="Sheng Gao",
    author_email="goseng123@gmail.com", 
    description="A package for parsing various document types.",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    url="https://github.com/aigaosheng/parse-any-doc", 
    packages=find_packages(where="src"),
    package_dir={"": "src"}, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11", 
    install_requires=[ 
        "paddleocr", 
        "pandas",
        "docx2txt",
    ],
    entry_points={  
        "console_scripts": [
            # "parse-any-doc=audit_logger.audit_logger:main",  # Example script
        ],
    },
)